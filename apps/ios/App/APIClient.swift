import Foundation
import MuralCore
import UIKit

final class NoRedirect: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
}

struct APIUsage { var input = 0; var output = 0; var searches = 0 }
struct APIResult { var text: String; var sources: [SourceLink]; var usage: APIUsage }

@MainActor final class APIClient {
    var conversationProvider: ConversationProvider = .personalKey
    var hostedLease: HostedLease?
    var canProcessAI: () -> Bool = { true }
    private var responseTasks: [UUID: Task<APIResult, Error>] = [:]
    func cancelAIRequests() {
        responseTasks.values.forEach { $0.cancel() }
        responseTasks.removeAll()
    }
    private let session: URLSession
    private var voiceCredential = VoiceCredentialScope()
    private var credentialExpiry: Task<Void, Never>?
    func beginVoiceCredential() {
        credentialExpiry?.cancel(); voiceCredential.clear()
        guard let key = CredentialStore.read() else { return }
        voiceCredential.begin(key: key)
        expireVoiceCredential(after: .seconds(65 * 60))
    }
    func endVoiceCredential() {
        voiceCredential.end()
        expireVoiceCredential(after: .seconds(60))
    }
    private func expireVoiceCredential(after duration: Duration) {
        credentialExpiry?.cancel()
        credentialExpiry = Task { [weak self] in
            do { try await Task.sleep(for: duration) } catch { return }
            self?.voiceCredential.clear(); self?.credentialExpiry = nil
        }
    }
    private func personalKey() -> String? {
        voiceCredential.credential(protectedDataAvailable: UIApplication.shared.isProtectedDataAvailable,
                                   readStored: CredentialStore.read)
    }
    /// A saved Google AI Studio key selects Gemini; an OpenAI key keeps the original path.
    func geminiKey() -> String? {
        guard conversationProvider == .personalKey, let key = personalKey(), CredentialStore.isGemini(key) else { return nil }
        return key
    }
    init() {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 45; config.timeoutIntervalForResource = 60
        config.httpCookieStorage = nil; config.urlCache = nil
        session = URLSession(configuration: config, delegate: NoRedirect(), delegateQueue: nil)
    }
    func post(_ path: String, body: [String: Any]) async throws -> [String: Any] {
        guard canProcessAI() else { throw AIProcessingConsent.ConsentError.required }
        guard let key = personalKey() else { throw APIError.missingKey }
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/" + path)!)
        request.httpMethod = "POST"; request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await session.data(for: request)
        try Task.checkCancellation()
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else { throw ProviderFailure(status: http.statusCode, body: data, reference: http.value(forHTTPHeaderField: "x-request-id")) }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw APIError.invalidResponse }
        return json
    }
    func respond(instructions: String, input: String, schema: [String: Any]? = nil, search: Bool = false,
                 purpose: String = "meaning", onText: (@MainActor (String) -> Void)? = nil) async throws -> APIResult {
        guard canProcessAI() else { throw AIProcessingConsent.ConsentError.required }
        try Task.checkCancellation()
        let id = UUID()
        let task = Task { @MainActor in
            try Task.checkCancellation()
            guard self.canProcessAI() else { throw AIProcessingConsent.ConsentError.required }
            let result = try await self.performResponse(instructions: instructions, input: input, schema: schema,
                                                       search: search, purpose: purpose, onText: onText)
            try Task.checkCancellation()
            guard self.canProcessAI() else { throw AIProcessingConsent.ConsentError.required }
            return result
        }
        responseTasks[id] = task
        defer { responseTasks.removeValue(forKey: id) }
        return try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
    }
    private func performResponse(instructions: String, input: String, schema: [String: Any]?, search: Bool,
                                 purpose: String, onText: (@MainActor (String) -> Void)?) async throws -> APIResult {
        if conversationProvider == .hosted {
            guard let hostedLease, let client = HostedClient.shared else { throw HostedError.unavailable }
            let result = try await client.helper(hostedLease, purpose: purpose, instructions: instructions, input: input,
                                                 schema: schema, search: search)
            onText?(result.text)
            return result
        }
        if let key = geminiKey() {
            let result = try await geminiResponse(key: key, instructions: instructions, input: input, schema: schema, search: search)
            onText?(result.text)
            return result
        }
        var body: [String: Any] = ["model": "gpt-6-luna", "store": false, "instructions": instructions,
                                  "input": [["role": "user", "content": input]], "max_output_tokens": schema == nil ? 1400 : 2200,
                                  "reasoning": ["effort": "low"]]
        if let schema { body["text"] = ["format": ["type": "json_schema", "name": "mural_result", "strict": true, "schema": schema]] }
        if search { body["tools"] = [["type": "web_search"]]; body["tool_choice"] = "auto"; body["max_tool_calls"] = 1 }
        let json: [String: Any]
        if let onText {
            body["stream"] = true
            json = try await streamResponse(body: body, onText: onText)
        } else { json = try await post("responses", body: body) }
        guard json["status"] as? String == "completed" else { throw APIError.incomplete }
        var text = "", sources: [SourceLink] = [], usage = APIUsage()
        for item in json["output"] as? [[String: Any]] ?? [] {
            if item["type"] as? String == "web_search_call" { usage.searches += 1 }
            for content in item["content"] as? [[String: Any]] ?? [] {
                if content["type"] as? String == "refusal" { throw APIError.refused }
                if content["type"] as? String == "output_text" { text += content["text"] as? String ?? "" }
                for citation in content["annotations"] as? [[String: Any]] ?? [] {
                    guard citation["type"] as? String == "url_citation", let url = citation["url"] as? String else { continue }
                    let source = SourceLink(title: citation["title"] as? String ?? "Source", url: url)
                    if source.safeURL != nil && !sources.contains(where: { $0.url == url }) { sources.append(source) }
                }
            }
        }
        if let u = json["usage"] as? [String: Any] { usage.input = u["input_tokens"] as? Int ?? 0; usage.output = u["output_tokens"] as? Int ?? 0 }
        guard !text.isEmpty else { throw APIError.incomplete }
        return APIResult(text: text, sources: sources, usage: usage)
    }
    /// Gemini 2.5 Flash: free tier, including daily Google Search grounding.
    private func geminiResponse(key: String, instructions: String, input: String, schema: [String: Any]?, search: Bool) async throws -> APIResult {
        var generation: [String: Any] = ["maxOutputTokens": schema == nil ? 1400 : 2200, "thinkingConfig": ["thinkingBudget": 0]]
        if let schema { generation["responseMimeType"] = "application/json"; generation["responseJsonSchema"] = schema }
        var body: [String: Any] = ["systemInstruction": ["parts": [["text": instructions]]],
                                   "contents": [["role": "user", "parts": [["text": input]]]],
                                   "generationConfig": generation]
        if search { body["tools"] = [["google_search": [String: Any]()]] }
        var request = URLRequest(url: URL(string: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")!)
        request.httpMethod = "POST"; request.setValue(key, forHTTPHeaderField: "x-goog-api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await session.data(for: request)
        try Task.checkCancellation()
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else { throw APIError.http(http.statusCode) }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw APIError.invalidResponse }
        if (json["promptFeedback"] as? [String: Any])?["blockReason"] != nil { throw APIError.refused }
        guard let candidate = (json["candidates"] as? [[String: Any]])?.first else { throw APIError.incomplete }
        if ["SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"].contains(candidate["finishReason"] as? String ?? "") { throw APIError.refused }
        var text = "", sources: [SourceLink] = [], usage = APIUsage()
        for part in (candidate["content"] as? [String: Any])?["parts"] as? [[String: Any]] ?? [] where part["thought"] as? Bool != true {
            text += part["text"] as? String ?? ""
        }
        if let grounding = candidate["groundingMetadata"] as? [String: Any] {
            if !(grounding["webSearchQueries"] as? [Any] ?? []).isEmpty { usage.searches += 1 }
            for chunk in grounding["groundingChunks"] as? [[String: Any]] ?? [] {
                guard let web = chunk["web"] as? [String: Any], let url = web["uri"] as? String else { continue }
                let source = SourceLink(title: web["title"] as? String ?? "Source", url: url)
                if source.safeURL != nil && !sources.contains(where: { $0.url == url }) { sources.append(source) }
            }
        }
        if let u = json["usageMetadata"] as? [String: Any] {
            usage.input = u["promptTokenCount"] as? Int ?? 0; usage.output = u["candidatesTokenCount"] as? Int ?? 0
        }
        guard !text.isEmpty else { throw APIError.incomplete }
        return APIResult(text: text, sources: sources, usage: usage)
    }
    private func streamResponse(body: [String: Any], onText: @MainActor (String) -> Void) async throws -> [String: Any] {
        guard canProcessAI() else { throw AIProcessingConsent.ConsentError.required }
        guard let key = personalKey() else { throw APIError.missingKey }
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/responses")!)
        request.httpMethod = "POST"
        request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            var body = Data()
            for try await byte in bytes { body.append(byte); if body.count >= 16_384 { break } }
            throw ProviderFailure(status: http.statusCode, body: body, reference: http.value(forHTTPHeaderField: "x-request-id"))
        }
        guard http.value(forHTTPHeaderField: "Content-Type")?.lowercased().hasPrefix("text/event-stream") == true else { throw APIError.invalidResponse }
        var decoder = ResponseTextStream()
        do {
            for try await byte in bytes {
                try Task.checkCancellation()
                guard let update = try decoder.consume(byte: byte) else { continue }
                switch update {
                case .text(let value): onText(value)
                case .completed(let result): return result
                }
            }
        } catch ResponseTextStream.Failure.refused { throw APIError.refused }
        catch is ResponseTextStream.Failure { throw APIError.incomplete }
        throw APIError.incomplete
    }
    static func object(_ fields: [String: Any]) -> [String: Any] { ["type": "object", "properties": fields, "required": fields.keys.sorted(), "additionalProperties": false] }
    static let string: [String: Any] = ["type": "string"]
    static func assessmentSchema(language: LanguageModule) -> [String: Any] { object([
        "outcome": ["type": "string", "enum": ["success", "partial", "breakdown", "uncertain"]],
        "suggestedLevel": ["type": "integer", "minimum": 0, "maximum": 5], "nextGoal": string, "capability": string,
        "words": ["type": "array", "maxItems": 12, "items": object([
            "lemma": string, "meaning": string, "form": string, "quote": string, "language": ["type": "string", "enum": Array(Set([language.id, "en", "mixed", "uncertain"])).sorted()],
            "kind": ["type": "string", "enum": ["exposure", "understanding", "assisted", "independent", "lapse"]],
            "confidence": ["type": "number", "minimum": 0, "maximum": 1], "sourceIDs": ["type": "array", "items": string]
        ])]
    ]) }
    enum APIError: LocalizedError {
        case missingKey, invalidResponse, incomplete, refused, http(Int)
        var errorDescription: String? {
            switch self {
            case .missingKey: "Add your Gemini or OpenAI key in Settings to begin."
            case .invalidResponse, .incomplete: "The AI service returned an incomplete response. Please try again."
            case .refused: "Mural couldn’t complete that request. Try a different topic."
            case .http(400), .http(401): "Your API key wasn’t accepted. Check it in Settings."
            case .http(403), .http(404): "This API key may not have access to the requested model."
            case .http(429): "The free usage limit was reached for now. Try again later."
            case .http(let status): "The AI service couldn’t complete the request (HTTP \(status)). Please try again."
            }
        }
    }
}
