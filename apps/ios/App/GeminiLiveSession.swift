import Foundation
import AVFoundation

/// Voice conversation over the Gemini Live API (WebSocket, PCM audio).
/// It emits the same `session.*` events as the OpenAI data channel, so
/// ConversationCoordinator works unchanged with either provider.
@MainActor final class GeminiLiveSession {
    static let model = "models/gemini-3.8-live"
    static let delegationTool = "ask_mural"
    var onEvent: (([String: Any]) -> Void)?
    var onLevels: ((Double, Double) -> Void)?
    var onFailure: ((String) -> Void)?
    private(set) var started = false
    private var socket: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let speakerFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)!
    private var startedAt = Date()
    private var closing = false
    private var muted = false
    private var failure: String?
    private var pendingCalls: [String: String] = [:]
    private var inputLevel = 0.0, outputLevel = 0.0
    private var playingUntil = Date.distantPast
    private var meterTask: Task<Void, Never>?

    private var elapsedMS: Int { max(0, Int(Date().timeIntervalSince(startedAt) * 1000)) }

    func connect(key: String, instructions: String, history: [[String: Any]]) async throws {
        let audio = AVAudioSession.sharedInstance()
        try audio.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try audio.setActive(true)

        var components = URLComponents(string: "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")!
        components.queryItems = [URLQueryItem(name: "key", value: key)]
        let config = URLSessionConfiguration.ephemeral
        config.httpCookieStorage = nil; config.urlCache = nil
        let session = URLSession(configuration: config)
        let socket = session.webSocketTask(with: components.url!)
        socket.maximumMessageSize = 16 * 1024 * 1024
        self.urlSession = session; self.socket = socket
        socket.resume()
        receive()

        var system = instructions + "\nWhen the learner asks for current facts, news, or a detailed explanation, call \(Self.delegationTool) and then speak its answer naturally. Notes that start with [Mural app] come from the app, not the learner: follow them silently and never read them aloud."
        let earlier = history.compactMap { item -> String? in
            guard let role = item["role"] as? String,
                  let text = (item["content"] as? [[String: Any]])?.first?["text"] as? String else { return nil }
            return (role == "user" ? "Learner: " : "You: ") + text
        }
        if !earlier.isEmpty { system += "\nEarlier in this conversation (continue from here):\n" + earlier.joined(separator: "\n") }
        let tool: [String: Any] = [
            "name": Self.delegationTool,
            "description": "Ask Mural's helper for current facts (with web search) or a detailed language explanation. Returns a short answer in the learning language.",
            "parameters": ["type": "object", "properties": ["request": ["type": "string", "description": "What the learner wants to know"]], "required": ["request"]]
        ]
        send(["setup": [
            "model": Self.model,
            "generationConfig": ["responseModalities": ["AUDIO"]],
            "systemInstruction": ["parts": [["text": system]]],
            "tools": [["functionDeclarations": [tool]]],
            "inputAudioTranscription": [String: Any](),
            "outputAudioTranscription": [String: Any](),
            "contextWindowCompression": ["slidingWindow": [String: Any]()]
        ]])

        let deadline = Date().addingTimeInterval(20)
        while !started {
            if let failure { stop(); throw GeminiError.rejected(failure) }
            try await Task.sleep(for: .milliseconds(100))
            guard Date() < deadline else { stop(); throw LiveTransport.TransportError.timeout }
        }
    }

    /// Maps the coordinator's OpenAI-style client events onto Gemini messages.
    @discardableResult func send(event: [String: Any]) -> Bool {
        guard started, let type = event["type"] as? String else { return false }
        let content = event["content"] as? String ?? ""
        switch type {
        case "session.commentary.append":
            if let id = event["delegation_id"] as? String, let name = pendingCalls.removeValue(forKey: id) {
                send(["toolResponse": ["functionResponses": [["id": id, "name": name, "response": ["result": content]]]]])
            } else { note(content, respond: true) }
        case "session.thinking.append": note(content, respond: false)
        case "session.instructions.append": note(content, respond: true)
        default: return false
        }
        if let id = event["event_id"] as? String {
            let appended = type.replacingOccurrences(of: ".append", with: ".appended")
            Task { @MainActor [weak self] in self?.onEvent?(["type": appended, "client_event_id": id]) }
        }
        return true
    }
    func mute(_ value: Bool) {
        muted = value
        if value { send(["realtimeInput": ["audioStreamEnd": true]]) }
    }
    func close() {
        closing = true; muted = true
        let seconds = Double(elapsedMS) / 1000
        Task { @MainActor [weak self] in
            self?.onEvent?(["type": "session.closed", "reason": "Ended by you", "usage": ["seconds": seconds]])
        }
    }
    func stop() {
        closing = true; started = false
        meterTask?.cancel(); meterTask = nil
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            player.stop(); engine.stop()
        }
        socket?.cancel(with: .normalClosure, reason: nil); socket = nil
        urlSession?.invalidateAndCancel(); urlSession = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        onLevels?(0, 0)
    }

    private func note(_ text: String, respond: Bool) {
        guard !text.isEmpty else { return }
        send(["clientContent": ["turns": [["role": "user", "parts": [["text": "[Mural app] " + text]]]], "turnComplete": respond]])
    }
    private func send(_ message: [String: Any]) {
        guard let socket, let data = try? JSONSerialization.data(withJSONObject: message),
              let text = String(data: data, encoding: .utf8) else { return }
        socket.send(.string(text)) { _ in }
    }

    private func receive() {
        socket?.receive { [weak self] result in
            Task { @MainActor [weak self] in
                guard let self else { return }
                switch result {
                case .success(let message):
                    let data: Data?
                    switch message {
                    case .string(let text): data = text.data(using: .utf8)
                    case .data(let bytes): data = bytes
                    @unknown default: data = nil
                    }
                    if let data, let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] { self.handle(json) }
                    self.receive()
                case .failure:
                    guard !self.closing else { return }
                    let reason = self.socket?.closeReason.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                    let message = GeminiError.describe(reason)
                    if self.started { self.onFailure?(message) } else { self.failure = message }
                }
            }
        }
    }

    private func handle(_ json: [String: Any]) {
        if json["setupComplete"] != nil {
            startedAt = Date()
            do { try startAudio() } catch { failure = "Mural couldn’t start the microphone. Try again."; return }
            started = true
            onEvent?(["type": "session.started", "session": ["id": "gemini-" + UUID().uuidString]])
            startMetering()
        }
        if let content = json["serverContent"] as? [String: Any] {
            if content["interrupted"] as? Bool == true { player.stop(); player.play(); playingUntil = .distantPast }
            for part in (content["modelTurn"] as? [String: Any])?["parts"] as? [[String: Any]] ?? [] {
                if let inline = part["inlineData"] as? [String: Any], let b64 = inline["data"] as? String,
                   let pcm = Data(base64Encoded: b64) { play(pcm) }
            }
            if let text = (content["inputTranscription"] as? [String: Any])?["text"] as? String, !text.isEmpty {
                transcript("session.input_transcript.delta", text)
            }
            if let text = (content["outputTranscription"] as? [String: Any])?["text"] as? String, !text.isEmpty {
                transcript("session.output_transcript.delta", text)
            }
        }
        if let calls = (json["toolCall"] as? [String: Any])?["functionCalls"] as? [[String: Any]] {
            for call in calls {
                guard let id = call["id"] as? String, let name = call["name"] as? String else { continue }
                pendingCalls[id] = name
                onEvent?(["type": "session.delegation.created", "delegation": ["id": id, "target": "client"]])
            }
        }
        if let ids = (json["toolCallCancellation"] as? [String: Any])?["ids"] as? [String] {
            ids.forEach { pendingCalls.removeValue(forKey: $0) }
        }
    }
    private func transcript(_ type: String, _ text: String) {
        let now = elapsedMS
        onEvent?(["type": type, "delta": text, "start_ms": now, "end_ms": now, "event_id": UUID().uuidString])
    }

    private func startAudio() throws {
        let input = engine.inputNode
        try input.setVoiceProcessingEnabled(true)
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: speakerFormat)
        let hardware = input.outputFormat(forBus: 0)
        let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)!
        guard let converter = AVAudioConverter(from: hardware, to: target) else { throw LiveTransport.TransportError.connection }
        let tap = Self.microphoneTap(converter: converter, target: target, inputRate: hardware.sampleRate) { [weak self] pcm, level in
            Task { @MainActor [weak self] in self?.microphone(pcm, level: level) }
        }
        input.installTap(onBus: 0, bufferSize: 2048, format: hardware, block: tap)
        engine.prepare()
        try engine.start()
        player.play()
    }
    /// Built outside the main actor: the tap runs on the audio thread.
    nonisolated private static func microphoneTap(converter: AVAudioConverter, target: AVAudioFormat, inputRate: Double,
                                                  deliver: @escaping @Sendable (Data, Double) -> Void) -> AVAudioNodeTapBlock {
        { buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * 16_000 / inputRate) + 64
            guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            var fed = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true; status.pointee = .haveData; return buffer
            }
            guard error == nil, out.frameLength > 0, let samples = out.int16ChannelData?[0] else { return }
            let count = Int(out.frameLength)
            var sum = 0.0
            for i in 0..<count { let s = Double(samples[i]) / 32768; sum += s * s }
            deliver(Data(bytes: samples, count: count * 2), sqrt(sum / Double(count)))
        }
    }
    private func microphone(_ pcm: Data, level: Double) {
        guard started, !closing, !muted else { inputLevel = 0; return }
        inputLevel = inputLevel * 0.35 + min(1, level * 6) * 0.65
        send(["realtimeInput": ["audio": ["data": pcm.base64EncodedString(), "mimeType": "audio/pcm;rate=16000"]]])
    }
    private func play(_ pcm: Data) {
        let count = pcm.count / 2
        guard count > 0, let buffer = AVAudioPCMBuffer(pcmFormat: speakerFormat, frameCapacity: AVAudioFrameCount(count)),
              let channel = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = AVAudioFrameCount(count)
        var sum = 0.0
        pcm.withUnsafeBytes { raw in
            let samples = raw.bindMemory(to: Int16.self)
            for i in 0..<count {
                let s = Float(Int16(littleEndian: samples[i])) / 32768
                channel[i] = s; sum += Double(s * s)
            }
        }
        outputLevel = min(1, sqrt(sum / Double(count)) * 6)
        player.scheduleBuffer(buffer)
        playingUntil = max(playingUntil, Date()).addingTimeInterval(Double(count) / 24_000)
    }
    private func startMetering() {
        meterTask?.cancel()
        meterTask = Task { [weak self] in
            var ticks = 0
            while !Task.isCancelled {
                guard let self else { return }
                let output = Date() < self.playingUntil ? self.outputLevel : 0
                self.onLevels?(self.muted ? 0 : self.inputLevel, output)
                self.inputLevel *= 0.8
                ticks += 1
                if ticks % 50 == 0 {
                    self.onEvent?(["type": "session.usage.updated", "usage": ["seconds": Double(self.elapsedMS) / 1000]])
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }

    enum GeminiError: LocalizedError {
        case rejected(String)
        var errorDescription: String? { if case .rejected(let message) = self { message } else { nil } }
        static func describe(_ reason: String) -> String {
            let lower = reason.lowercased()
            if lower.contains("api key") || lower.contains("permission") || lower.contains("unauth") {
                return "Your Gemini key wasn’t accepted. Check it in Settings → Advanced."
            }
            if lower.contains("quota") || lower.contains("exhausted") || lower.contains("rate") {
                return "Gemini’s free limit is used up for now. Try again later or tomorrow."
            }
            if lower.contains("model") && lower.contains("not") {
                return "This Gemini voice model isn’t available for your key."
            }
            return reason.isEmpty ? "The Gemini voice connection ended unexpectedly. Your conversation has been saved."
                                  : "Gemini ended the conversation: \(reason.prefix(160))"
        }
    }
}
