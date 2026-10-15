// http_server.hpp - minimal HTTP/1.1 plumbing for the embedded Web UI.
// Plain sockets, sequential request handling (single local user). Extracted
// from the moondream server so every backend's UI shares one implementation.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <winsock2.h>
  #include <ws2tcpip.h>
#else
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <unistd.h>
#endif

namespace vqa {
namespace http {
#ifdef _WIN32
typedef SOCKET HttpSock;
#define HTTP_INVALID INVALID_SOCKET
#define http_close closesocket
#else
typedef int HttpSock;
#define HTTP_INVALID (-1)
#define http_close close
#endif

struct Request {
    std::string method, path, query;   // path excludes '?'; query excludes '?'
    std::map<std::string, std::string> headers;   // lowercased keys
    std::string body;
};

inline void url_decode(std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            auto hex = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
            out += (char)(hex(s[i + 1]) * 16 + hex(s[i + 2]));
            i += 2;
        } else if (s[i] == '+') {
            out += ' ';
        } else {
            out += s[i];
        }
    }
    s.swap(out);
}

inline std::string query_param(const std::string & q, const char * key) {
    std::string pat = std::string(key) + "=";
    size_t p = q.find(pat);
    if (p == std::string::npos) return "";
    size_t e = q.find('&', p);
    std::string v = q.substr(p + pat.size(), e == std::string::npos ? std::string::npos : e - p - pat.size());
    url_decode(v);
    return v;
}

inline std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
                else out += c;
        }
    }
    return out;
}

inline void send(HttpSock s, const std::string & status, const std::string & ctype, const std::string & body) {
    std::string h = "HTTP/1.1 " + status + "\r\nContent-Type: " + ctype +
                    "\r\nContent-Length: " + std::to_string(body.size()) +
                    "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    ::send(s, h.data(), (int)h.size(), 0);
    ::send(s, body.data(), (int)body.size(), 0);
}

inline void send_json(HttpSock s, const std::string & body) { send(s, "200 OK", "application/json", body); }
inline void send_error(HttpSock s, const std::string & msg) {
    send(s, "400 Bad Request", "application/json", "{\"error\":\"" + json_escape(msg) + "\"}");
}

// --- chunked streaming (Connection: close; one chunk per event) ---
inline void send_stream_head(HttpSock s, const char * ctype) {
    std::string h = std::string("HTTP/1.1 200 OK\r\nContent-Type: ") + ctype +
                    "\r\nTransfer-Encoding: chunked\r\nCache-Control: no-store"
                    "\r\nX-Accel-Buffering: no\r\nConnection: close\r\n\r\n";
    ::send(s, h.data(), (int)h.size(), 0);
}

// returns false when the peer went away (browser tab closed / aborted fetch)
inline bool send_chunk_str(HttpSock s, const std::string & data) {
    char head[16];
    snprintf(head, sizeof(head), "%zx\r\n", data.size());
    std::string chunk = std::string(head) + data + "\r\n";
    return ::send(s, chunk.data(), (int)chunk.size(), 0) > 0;
}

inline void send_stream_end(HttpSock s) { ::send(s, "0\r\n\r\n", 5, 0); }

inline bool recv_all(HttpSock s, Request & req) {
    std::string buf;
    char tmp[8192];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        int n = ::recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, n);
        hdr_end = buf.find("\r\n\r\n");
        if (buf.size() > (1u << 20)) return false;
    }
    std::string head = buf.substr(0, hdr_end);
    size_t sp = head.find(' ');
    req.method = head.substr(0, sp);
    std::string target = head.substr(sp + 1, head.find(' ', sp + 1) - sp - 1);
    size_t q = target.find('?');
    req.path = q == std::string::npos ? target : target.substr(0, q);
    req.query = q == std::string::npos ? "" : target.substr(q + 1);
    url_decode(req.path);
    // headers
    size_t pos = head.find("\r\n") + 2;
    while (pos < head.size()) {
        size_t e = head.find("\r\n", pos);
        if (e == std::string::npos) e = head.size();
        std::string line = head.substr(pos, e - pos);
        size_t c = line.find(':');
        if (c != std::string::npos) {
            std::string k = line.substr(0, c);
            for (auto & ch : k) ch = (char)tolower((unsigned char)ch);
            req.headers[k] = line.substr(c + 2);
        }
        pos = e + 2;
    }
    size_t clen = 0;
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) clen = strtoull(it->second.c_str(), nullptr, 10);
    req.body = buf.substr(hdr_end + 4);
    while (req.body.size() < clen) {
        int n = ::recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        req.body.append(tmp, n);
    }
    req.body.resize(std::min(req.body.size(), clen));
    return true;
}

// one-shot listener; handler returns false only on fatal errors
using Handler = bool (*)(HttpSock, const Request &);

inline int serve(int port, Handler handler, const char * bind_host = "127.0.0.1") {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { fprintf(stderr, "WSAStartup failed\n"); return 1; }
#endif
    HttpSock srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv == HTTP_INVALID) { fprintf(stderr, "socket failed\n"); return 1; }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = bind_host && strcmp(bind_host, "0.0.0.0") == 0
        ? htonl(INADDR_ANY) : inet_addr(bind_host ? bind_host : "127.0.0.1");
    addr.sin_port = htons((u_short)port);
    if (::bind(srv, (sockaddr *)&addr, sizeof(addr)) != 0) { fprintf(stderr, "bind failed on port %d\n", port); return 1; }
    if (::listen(srv, 4) != 0) { fprintf(stderr, "listen failed\n"); return 1; }
    while (true) {
        sockaddr_in cli {};
#ifdef _WIN32
        int cl = sizeof(cli);
#else
        socklen_t cl = sizeof(cli);
#endif
        HttpSock c = ::accept(srv, (sockaddr *)&cli, &cl);
        if (c == HTTP_INVALID) continue;
        Request req;
        if (!recv_all(c, req)) { http_close(c); continue; }
        handler(c, req);
        http_close(c);
    }
    return 0;
}

}  // namespace http
}  // namespace vqa
