// A minimal JSON value, parser and serializer: just enough for the onnxsim RPC
// headers (objects, arrays, strings, numbers, bools, null). Not a
// general-purpose library.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Json {
  enum class Type { Null, Bool, Int, Double, String, Array, Object };
  Type type = Type::Null;
  bool b = false;
  int64_t i = 0;
  double d = 0;
  std::string s;
  std::vector<Json> a;
  std::vector<std::pair<std::string, Json>> o;

  Json() = default;
  static Json Bool(bool v) {
    Json j;
    j.type = Type::Bool;
    j.b = v;
    return j;
  }
  static Json Int(int64_t v) {
    Json j;
    j.type = Type::Int;
    j.i = v;
    return j;
  }
  static Json Double(double v) {
    Json j;
    j.type = Type::Double;
    j.d = v;
    return j;
  }
  static Json String(std::string v) {
    Json j;
    j.type = Type::String;
    j.s = std::move(v);
    return j;
  }
  static Json Array() {
    Json j;
    j.type = Type::Array;
    return j;
  }
  static Json Object() {
    Json j;
    j.type = Type::Object;
    return j;
  }

  Json& set(const std::string& key, Json value) {
    for (auto& kv : o)
      if (kv.first == key) {
        kv.second = std::move(value);
        return *this;
      }
    o.emplace_back(key, std::move(value));
    return *this;
  }
  Json& push(Json value) {
    a.push_back(std::move(value));
    return *this;
  }
  const Json* find(const std::string& key) const {
    for (const auto& kv : o)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  bool has(const std::string& key) const {
    auto* v = find(key);
    return v && v->type != Type::Null;
  }
  std::string str(const std::string& key,
                  const std::string& fallback = "") const {
    auto* v = find(key);
    return (v && v->type == Type::String) ? v->s : fallback;
  }
  int64_t integer(const std::string& key, int64_t fallback = 0) const {
    auto* v = find(key);
    if (!v) return fallback;
    if (v->type == Type::Int) return v->i;
    if (v->type == Type::Double) return static_cast<int64_t>(v->d);
    if (v->type == Type::Bool) return v->b ? 1 : 0;
    return fallback;
  }
  bool flag(const std::string& key) const { return integer(key, 0) != 0; }

  std::string dump() const {
    std::string out;
    write(out);
    return out;
  }

  static Json parse(const std::string& text) {
    size_t pos = 0;
    Json value = parse_value(text, pos);
    skip(text, pos);
    if (pos != text.size())
      throw std::runtime_error("trailing characters in JSON");
    return value;
  }

 private:
  static void skip(const std::string& t, size_t& p) {
    while (p < t.size() &&
           (t[p] == ' ' || t[p] == '\n' || t[p] == '\r' || t[p] == '\t'))
      ++p;
  }
  static void write_string(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
      switch (c) {
        case '"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
          } else
            out += static_cast<char>(c);
      }
    }
    out += '"';
  }
  void write(std::string& out) const {
    switch (type) {
      case Type::Null:
        out += "null";
        break;
      case Type::Bool:
        out += b ? "true" : "false";
        break;
      case Type::Int:
        out += std::to_string(i);
        break;
      case Type::Double: {
        if (!std::isfinite(d)) {
          out += "null";
          break;
        }
        char buf[40];
        std::snprintf(buf, sizeof buf, "%.17g", d);
        out += buf;
        break;
      }
      case Type::String:
        write_string(out, s);
        break;
      case Type::Array:
        out += '[';
        for (size_t k = 0; k < a.size(); ++k) {
          if (k) out += ',';
          a[k].write(out);
        }
        out += ']';
        break;
      case Type::Object:
        out += '{';
        for (size_t k = 0; k < o.size(); ++k) {
          if (k) out += ',';
          write_string(out, o[k].first);
          out += ':';
          o[k].second.write(out);
        }
        out += '}';
        break;
    }
  }
  static std::string parse_string(const std::string& t, size_t& p) {
    if (t[p] != '"') throw std::runtime_error("expected string");
    ++p;
    std::string out;
    while (p < t.size() && t[p] != '"') {
      char c = t[p++];
      if (c != '\\') {
        out += c;
        continue;
      }
      if (p >= t.size()) break;
      char e = t[p++];
      switch (e) {
        case '"':
          out += '"';
          break;
        case '\\':
          out += '\\';
          break;
        case '/':
          out += '/';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          if (p + 4 > t.size()) throw std::runtime_error("bad \\u escape");
          unsigned cp = static_cast<unsigned>(
              std::strtoul(t.substr(p, 4).c_str(), nullptr, 16));
          p += 4;
          if (cp < 0x80)
            out += static_cast<char>(cp);
          else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default:
          throw std::runtime_error("bad escape");
      }
    }
    if (p >= t.size()) throw std::runtime_error("unterminated string");
    ++p;
    return out;
  }
  static Json parse_value(const std::string& t, size_t& p) {
    skip(t, p);
    if (p >= t.size()) throw std::runtime_error("unexpected end of JSON");
    char c = t[p];
    if (c == '{') {
      ++p;
      Json j = Object();
      skip(t, p);
      if (t[p] == '}') {
        ++p;
        return j;
      }
      while (true) {
        skip(t, p);
        std::string key = parse_string(t, p);
        skip(t, p);
        if (t[p++] != ':') throw std::runtime_error("expected ':'");
        j.set(key, parse_value(t, p));
        skip(t, p);
        if (t[p] == ',') {
          ++p;
          continue;
        }
        if (t[p] == '}') {
          ++p;
          return j;
        }
        throw std::runtime_error("expected ',' or '}'");
      }
    }
    if (c == '[') {
      ++p;
      Json j = Array();
      skip(t, p);
      if (t[p] == ']') {
        ++p;
        return j;
      }
      while (true) {
        j.push(parse_value(t, p));
        skip(t, p);
        if (t[p] == ',') {
          ++p;
          continue;
        }
        if (t[p] == ']') {
          ++p;
          return j;
        }
        throw std::runtime_error("expected ',' or ']'");
      }
    }
    if (c == '"') return String(parse_string(t, p));
    if (t.compare(p, 4, "true") == 0) {
      p += 4;
      return Bool(true);
    }
    if (t.compare(p, 5, "false") == 0) {
      p += 5;
      return Bool(false);
    }
    if (t.compare(p, 4, "null") == 0) {
      p += 4;
      return Json();
    }
    size_t start = p;
    bool is_double = false;
    while (p < t.size() &&
           (std::isdigit(static_cast<unsigned char>(t[p])) || t[p] == '-' ||
            t[p] == '+' || t[p] == '.' || t[p] == 'e' || t[p] == 'E')) {
      if (t[p] == '.' || t[p] == 'e' || t[p] == 'E') is_double = true;
      ++p;
    }
    if (start == p) throw std::runtime_error("unexpected character in JSON");
    std::string num = t.substr(start, p - start);
    return is_double ? Double(std::strtod(num.c_str(), nullptr))
                     : Int(std::strtoll(num.c_str(), nullptr, 10));
  }
};
