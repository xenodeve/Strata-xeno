#pragma once
// A bounded Python subset for the programs a model writes inside its reasoning (#279): source to bytecode to a stack machine.
// Python calls live on heap frames, so a program's recursion never reaches the C++ stack and generator expressions can
// suspend. The C++ recursion that is left (parser, repr, compare, hash, a native calling back) is counted and measured.
// run() needs about 1 MiB of thread stack. No code here reaches files, network, processes, clock, environment or randomness.
#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::igc::sandbox {
enum class Status { Ok, Raised, Syntax, Unsupported, StepLimit, MemoryLimit, RecursionLimit, Cancelled };
struct Limits {
    uint64_t steps = 2'000'000;
    std::size_t heap_bytes = 64 * 1024 * 1024;
    std::size_t output_bytes = 4096;
    int recursion_depth = 200;
    int parser_depth = 100;
    int integer_digits = 4300;
};
struct Result {
    Status status = Status::Ok;
    std::string output;
    bool truncated = false;
    std::string error;
    uint64_t steps = 0;
    std::size_t peak_bytes = 0;
};
inline Result run(std::string_view source, const Limits& limits = {}, const std::atomic<bool>* cancel = nullptr);

namespace detail {
inline constexpr std::size_t kStackBudget = 640 * 1024;  // of C++ stack, measured from run()
inline constexpr std::size_t kValueStack = 1 << 14;      // slots of the Python stack, zeroed at every run

// Ends the run. Nothing in the program can catch it.
struct Stop {
    Status status;
    std::string error;
};

struct Obj;
struct Runtime {
    Limits limits;
    const std::atomic<bool>* cancel = nullptr;
    uint64_t steps = 0, next_poll = 0;
    std::size_t used = 0, peak = 0;
    Obj* all = nullptr;
    Obj* dying = nullptr;
    bool draining = false;
    std::uintptr_t stack_base = 0;
    int nesting = 0;
    std::string output;
    bool truncated = false;

    void charge(std::size_t n) {
        if (n > limits.heap_bytes - used) throw Stop{Status::MemoryLimit, "MemoryError: heap limit " + std::to_string(limits.heap_bytes) + " bytes"};
        used += n;
        if (used > peak) peak = used;
    }
    void release(std::size_t n) noexcept { used -= n; }
    void poll() {
        if (steps > limits.steps) throw Stop{Status::StepLimit, "StepLimit: " + std::to_string(limits.steps) + " steps"};
        if (cancel && cancel->load(std::memory_order_relaxed)) throw Stop{Status::Cancelled, "Cancelled"};
        next_poll = std::min(steps + 1024, limits.steps + 1);
    }
    void tick(uint64_t n = 1) {
        steps += n;
        if (steps >= next_poll) poll();
    }
};
inline thread_local Runtime* rt = nullptr;

inline std::size_t stack_used() {
    const char here = 0;
    const std::uintptr_t at = (std::uintptr_t) &here;
    return (std::size_t) (at < rt->stack_base ? rt->stack_base - at : at - rt->stack_base);
}
inline void check_stack() {
    if (stack_used() > kStackBudget) throw Stop{Status::RecursionLimit, "RecursionError: C++ stack budget"};
}
// One level of recursion over the program's data or back through a native.
struct Nest {
    Nest() {
        if (rt->nesting >= rt->limits.recursion_depth) throw Stop{Status::RecursionLimit, "RecursionError: depth limit " + std::to_string(rt->limits.recursion_depth)};
        check_stack();
        ++rt->nesting;
    }
    ~Nest() { --rt->nesting; }
    Nest(const Nest&) = delete;
    Nest& operator=(const Nest&) = delete;
};

// Charged before the allocation happens, so a refused size never reaches the allocator.
template <class T>
struct Alloc {
    using value_type = T;
    Alloc() = default;
    template <class U>
    Alloc(const Alloc<U>&) {}
    T* allocate(std::size_t n) {
        if (n > rt->limits.heap_bytes / sizeof(T) + 1) rt->charge(rt->limits.heap_bytes + 1);
        rt->charge(n * sizeof(T));
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }
    void deallocate(T* p, std::size_t n) noexcept {
        rt->release(n * sizeof(T));
        ::operator delete(p);
    }
    template <class U>
    bool operator==(const Alloc<U>&) const { return true; }
};

enum class Tag : uint8_t {
    Empty, None, Bool, Int, Float, Type, Native, Module, Stub, Ellipsis, Block,
    Big, Str, List, Tuple, Dict, Set, Range, Slice, Function, Bound, Iter, Exc, Cell, Cached, Fraction, Deque, View, Class, Instance, Method
};

// clang-format off
#define SANDBOX_TYPES(X) \
    X(object, object) X(NoneType, object) X(int, object) X(bool, int) X(float, object) X(str, object) X(list, object) \
    X(tuple, object) X(dict, object) X(set, object) X(frozenset, object) X(range, object) X(slice, object) \
    X(function, object) X(builtin_function_or_method, object) X(method, object) X(module, object) X(type, object) \
    X(generator, object) X(map, object) X(filter, object) X(zip, object) X(enumerate, object) X(reversed, object) \
    X(iterator, object) X(dict_keys, object) X(dict_values, object) X(dict_items, object) X(ellipsis, object) \
    X(Counter, dict) X(defaultdict, dict) X(deque, object) X(Fraction, object) X(_lru_cache_wrapper, object) \
    X(permutations, object) X(combinations, object) X(combinations_with_replacement, object) X(product, object) \
    X(accumulate, object) X(chain, object) X(count, object) X(islice, object) \
    X(BaseException, object) X(Exception, BaseException) X(GeneratorExit, BaseException) \
    X(KeyboardInterrupt, BaseException) X(SystemExit, BaseException) X(ArithmeticError, Exception) \
    X(FloatingPointError, ArithmeticError) X(OverflowError, ArithmeticError) X(ZeroDivisionError, ArithmeticError) \
    X(AssertionError, Exception) X(AttributeError, Exception) X(BufferError, Exception) X(EOFError, Exception) \
    X(ImportError, Exception) X(ModuleNotFoundError, ImportError) X(LookupError, Exception) X(IndexError, LookupError) \
    X(KeyError, LookupError) X(MemoryError, Exception) X(NameError, Exception) X(UnboundLocalError, NameError) \
    X(OSError, Exception) X(ReferenceError, Exception) X(RuntimeError, Exception) X(NotImplementedError, RuntimeError) \
    X(RecursionError, RuntimeError) X(StopIteration, Exception) X(SyntaxError, Exception) X(IndentationError, SyntaxError) \
    X(TabError, IndentationError) X(SystemError, Exception) X(TypeError, Exception) X(ValueError, Exception) \
    X(UnicodeError, ValueError) X(UnicodeDecodeError, UnicodeError) X(UnicodeEncodeError, UnicodeError) \
    X(Warning, Exception) X(UserWarning, Warning) X(DeprecationWarning, Warning) X(RuntimeWarning, Warning)
// clang-format on
enum TypeId : int {
#define X(name, base) T_##name,
    SANDBOX_TYPES(X)
#undef X
    kTypeCount
};
struct TypeInfo {
    const char* name;
    int base;
};
inline constexpr TypeInfo kTypes[] = {
#define X(name, base) {#name, T_##base},
    SANDBOX_TYPES(X)
#undef X
};
inline bool subtype(int type, int of) {
    for (;; type = kTypes[type].base) {
        if (type == of) return true;
        if (type == T_object) return false;
    }
}

struct Obj {
    uint32_t rc = 0;
    Tag tag;
    bool busy = false;   // repr is inside this object
    Obj* prev = nullptr;
    Obj* next = nullptr;
    explicit Obj(Tag t) : tag(t), next(rt->all) {
        if (next) next->prev = this;
        rt->all = this;
    }
    Obj(const Obj&) = delete;
    Obj& operator=(const Obj&) = delete;
    virtual ~Obj() = default;
    virtual void clear() {}
    static void* operator new(std::size_t n) {
        rt->charge(n);
        return ::operator new(n);
    }
    static void operator delete(void* p, std::size_t n) noexcept {
        rt->release(n);
        ::operator delete(p);
    }
};
// A deep list dies through this worklist: its destructor must not recurse once per level.
inline void destroy(Obj* o) noexcept {
    if (o->prev) o->prev->next = o->next;
    else rt->all = o->next;
    if (o->next) o->next->prev = o->prev;
    o->next = rt->dying;
    rt->dying = o;
    if (rt->draining) return;
    rt->draining = true;
    while (rt->dying) {
        Obj* x = rt->dying;
        rt->dying = x->next;
        delete x;
    }
    rt->draining = false;
}

struct NativeDef;
class Value {
public:
    Tag tag = Tag::Empty;
    uint32_t aux = 0;   // a Block's count of handled exceptions
    union {
        bool b;
        int64_t i = 0;
        double d;
        Obj* o;
        const NativeDef* native;
    };
    Value() noexcept {}
    explicit Value(Obj* obj) noexcept : tag(obj->tag), o(obj) { ++o->rc; }
    Value(const Value& v) noexcept : tag(v.tag), aux(v.aux), i(v.i) {
        if (tag >= Tag::Big) ++o->rc;
    }
    Value(Value&& v) noexcept : tag(v.tag), aux(v.aux), i(v.i) { v.tag = Tag::Empty; }
    Value& operator=(const Value& v) noexcept {
        if (v.tag >= Tag::Big) ++v.o->rc;
        drop();
        tag = v.tag;
        aux = v.aux;
        i = v.i;
        return *this;
    }
    Value& operator=(Value&& v) noexcept {
        if (this == &v) return *this;
        drop();
        tag = v.tag;
        aux = v.aux;
        i = v.i;
        v.tag = Tag::Empty;
        return *this;
    }
    ~Value() { drop(); }
    void reset() noexcept {
        drop();
        tag = Tag::Empty;
    }
    template <class T>
    T* as() const { return static_cast<T*>(o); }
    bool empty() const { return tag == Tag::Empty; }
    bool is_none() const { return tag == Tag::None; }
    static Value none() { Value v; v.tag = Tag::None; return v; }
    static Value boolean(bool x) { Value v; v.tag = Tag::Bool; v.b = x; v.aux = 0; return v; }
    static Value integer(int64_t x) { Value v; v.tag = Tag::Int; v.i = x; return v; }
    static Value real(double x) { Value v; v.tag = Tag::Float; v.d = x; return v; }
    static Value type(int id) { Value v; v.tag = Tag::Type; v.i = id; return v; }
    static Value module(int id) { Value v; v.tag = Tag::Module; v.i = id; return v; }
    static Value of(Tag t) { Value v; v.tag = t; return v; }
    static Value from(const NativeDef* n) { Value v; v.tag = Tag::Native; v.native = n; return v; }
private:
    void drop() noexcept {
        if (tag >= Tag::Big && --o->rc == 0) destroy(o);
    }
};
static_assert(sizeof(Value) == 16);

using Vals = std::vector<Value, Alloc<Value>>;
using U32 = std::basic_string<char32_t, std::char_traits<char32_t>, Alloc<char32_t>>;

struct StrObj : Obj {
    U32 s;
    uint64_t hash = 0;
    StrObj() : Obj(Tag::Str) {}
};
struct ListObj : Obj {
    Vals v;
    ListObj() : Obj(Tag::List) {}
    void clear() override { v.clear(); }
};
struct TupleObj : Obj {
    Vals v;
    TupleObj() : Obj(Tag::Tuple) {}
    void clear() override { v.clear(); }
};
struct Entry {
    uint64_t hash;
    Value key;
    Value val;
};
// Insertion ordered: `entries` in order, `index` an open-addressed table of positions. A deleted entry keeps its place
// with an empty key until the next rebuild.
struct Table {
    std::vector<Entry, Alloc<Entry>> entries;
    std::vector<int32_t, Alloc<int32_t>> index;
    std::size_t used = 0;
    uint64_t rebuilds = 0;
    static constexpr int32_t kFree = -1, kDummy = -2;
    int32_t find(const Value& key, uint64_t hash, std::size_t* slot = nullptr) const;
    void rebuild(std::size_t want);
    Value* get(const Value& key);
    void set(const Value& key, Value val);
    bool erase(const Value& key, Value* out = nullptr);
    void wipe() {
        entries.clear();
        index.clear();
        used = 0;
    }
};
enum class DictKind : uint8_t { Plain, Counter, Default };
struct DictObj : Obj {
    Table t;
    DictKind kind = DictKind::Plain;
    Value factory;
    DictObj() : Obj(Tag::Dict) {}
    void clear() override {
        t.wipe();
        factory.reset();
    }
};
struct SetObj : Obj {
    Table t;
    bool frozen = false;
    SetObj() : Obj(Tag::Set) {}
    void clear() override { t.wipe(); }
};
struct RangeObj : Obj {
    int64_t start, stop, step, len;
    RangeObj(int64_t a, int64_t b, int64_t c, int64_t n) : Obj(Tag::Range), start(a), stop(b), step(c), len(n) {}
};
struct SliceObj : Obj {
    Value start, stop, step;
    SliceObj() : Obj(Tag::Slice) {}
    void clear() override { start.reset(); stop.reset(); step.reset(); }
};
struct CellObj : Obj {
    Value v;
    CellObj() : Obj(Tag::Cell) {}
    void clear() override { v.reset(); }
};
struct Code;
struct FunctionObj : Obj {
    const Code* code;
    Vals defaults;       // of the last positional parameters
    Vals kw_defaults;    // one per keyword-only parameter, empty where it has none
    Vals closure;
    explicit FunctionObj(const Code* c) : Obj(Tag::Function), code(c) {}
    void clear() override { defaults.clear(); kw_defaults.clear(); closure.clear(); }
};
struct BoundObj : Obj {
    Value self;
    const NativeDef* method;
    BoundObj(Value s, const NativeDef* m) : Obj(Tag::Bound), self(std::move(s)), method(m) {}
    void clear() override { self.reset(); }
};
// A class holds methods and nothing else: `class X:` with `def`s and `__init__`, which covers what the model was seen to write (29 of
// 241 blocks had a class, each a `class Solution` of plain methods). The parser refuses the rest, so no protocol method is ever silently ignored.
using Attrs = std::vector<std::pair<std::string_view, Value>, Alloc<std::pair<std::string_view, Value>>>;
inline const Value* find_attr(const Attrs& attrs, std::string_view name) {
    for (std::size_t i = attrs.size(); i-- > 0;)
        if (attrs[i].first == name) return &attrs[i].second;
    return nullptr;
}
struct ClassObj : Obj {
    std::string_view name;
    Attrs methods;
    explicit ClassObj(std::string_view n) : Obj(Tag::Class), name(n) {}
    void clear() override { methods.clear(); }
};
struct InstanceObj : Obj {
    Value cls;
    Attrs attrs;
    explicit InstanceObj(Value c) : Obj(Tag::Instance), cls(std::move(c)) {}
    void clear() override { cls.reset(); attrs.clear(); }
};
struct MethodObj : Obj {
    Value self, fn;
    MethodObj(Value s, Value f) : Obj(Tag::Method), self(std::move(s)), fn(std::move(f)) {}
    void clear() override { self.reset(); fn.reset(); }
};
struct ExcObj : Obj {
    int type;
    Vals args;
    int line = 0;
    explicit ExcObj(int t) : Obj(Tag::Exc), type(t) {}
    void clear() override { args.clear(); }
};
struct IterObj : Obj {
    int type;
    explicit IterObj(int t = T_iterator) : Obj(Tag::Iter), type(t) {}
    virtual bool next(Value& out) = 0;
};
// functools.lru_cache: `t` in recency order, the oldest entry first.
struct CachedObj : Obj {
    Value fn;
    int64_t maxsize;
    Table t;
    int64_t hits = 0, misses = 0;
    CachedObj(Value f, int64_t m) : Obj(Tag::Cached), fn(std::move(f)), maxsize(m) {}
    void clear() override { fn.reset(); t.wipe(); }
};
struct FractionObj : Obj {
    Value num, den;
    FractionObj(Value n, Value d) : Obj(Tag::Fraction), num(std::move(n)), den(std::move(d)) {}
    void clear() override { num.reset(); den.reset(); }
};
struct DequeObj : Obj {
    Vals ring;
    std::size_t head = 0, count = 0;
    int64_t maxlen = -1;
    uint64_t version = 0;
    DequeObj() : Obj(Tag::Deque) {}
    void clear() override { ring.clear(); head = count = 0; }
    Value& at(std::size_t i) { return ring[(head + i) % ring.size()]; }
    void grow() {
        if (count < ring.size()) return;
        Vals fresh((std::max<std::size_t>)(8, ring.size() * 2));
        for (std::size_t i = 0; i < count; ++i) fresh[i] = std::move(at(i));
        ring.swap(fresh);
        head = 0;
    }
};
struct ViewObj : Obj {
    Value dict;
    int type;
    ViewObj(Value d, int t) : Obj(Tag::View), dict(std::move(d)), type(t) {}
    void clear() override { dict.reset(); }
};

struct PyError {
    Value exc;
};
struct Call {
    Value* a;
    int n;                       // positional arguments, `self` first for a method
    const TupleObj* kw_names;    // their values follow the positional ones
    int nkw() const { return kw_names ? (int) kw_names->v.size() : 0; }
    Value* keyword(const char* name) const;
};
using NativeFn = Value (*)(Call&);
struct NativeDef {
    const char* name;
    NativeFn fn;
    int min, max;     // positional arguments, -1 no bound
    bool keywords = false;
    bool is_static = false;
};

inline Value text(U32&& s) {
    auto* o = new StrObj;
    Value v(o);
    o->s = std::move(s);
    return v;
}
inline bool decode_utf8(std::string_view s, U32& out) {
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        if (c < 0x80) {
            out.push_back(c);
            ++i;
            continue;
        }
        const int more = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
        if (c < 0xC2 || c > 0xF4 || i + (std::size_t) more >= s.size()) return false;
        char32_t cp = more == 3 ? c & 7u : more == 2 ? c & 15u : c & 31u;
        for (int k = 1; k <= more; ++k) {
            const unsigned char t = (unsigned char) s[i + (std::size_t) k];
            if ((t & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (t & 63u);
        }
        if ((more == 2 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) || (more == 3 && (cp < 0x10000 || cp > 0x10FFFF))) return false;
        out.push_back(cp);
        i += (std::size_t) more + 1;
    }
    return true;
}
inline Value text(std::string_view utf8) {
    U32 s;
    decode_utf8(utf8, s);
    return text(std::move(s));
}
inline void put_utf8(std::string& out, char32_t c) {
    if (c < 0x80) out.push_back((char) c);
    else if (c < 0x800) {
        out.push_back((char) (0xC0 | (c >> 6)));
        out.push_back((char) (0x80 | (c & 63)));
    } else if (c < 0x10000) {
        out.push_back((char) (0xE0 | (c >> 12)));
        out.push_back((char) (0x80 | ((c >> 6) & 63)));
        out.push_back((char) (0x80 | (c & 63)));
    } else {
        out.push_back((char) (0xF0 | (c >> 18)));
        out.push_back((char) (0x80 | ((c >> 12) & 63)));
        out.push_back((char) (0x80 | ((c >> 6) & 63)));
        out.push_back((char) (0x80 | (c & 63)));
    }
}
inline std::string utf8(const U32& s) {
    std::string out;
    out.reserve(s.size());
    for (const char32_t c : s) put_utf8(out, c);
    return out;
}
inline void append(U32& out, std::string_view ascii) {
    for (const char c : ascii) out.push_back((unsigned char) c);
}
inline const U32& str_of(const Value& v) { return v.as<StrObj>()->s; }
inline bool equals(const Value& v, std::string_view ascii) {
    if (v.tag != Tag::Str) return false;
    const U32& s = str_of(v);
    if (s.size() != ascii.size()) return false;
    for (std::size_t i = 0; i < s.size(); ++i)
        if (s[i] != (unsigned char) ascii[i]) return false;
    return true;
}
inline Value* Call::keyword(const char* name) const {
    for (int k = 0; k < nkw(); ++k)
        if (equals(kw_names->v[(std::size_t) k], name)) return &a[n + k];
    return nullptr;
}

[[noreturn]] inline void raise(int type, std::string_view message) {
    auto* e = new ExcObj(type);
    Value holder(e);
    e->args.push_back(text(message));
    throw PyError{std::move(holder)};
}
[[noreturn]] inline void raise_value(int type, Value arg) {
    auto* e = new ExcObj(type);
    Value holder(e);
    e->args.push_back(std::move(arg));
    throw PyError{std::move(holder)};
}
[[noreturn]] inline void refuse(const std::string& what) { throw Stop{Status::Unsupported, "unsupported: " + what}; }

// Defined further down, used by the natives above them.
inline int type_of(const Value& v);
inline bool truth(const Value& v);
inline bool eq(const Value& a, const Value& b);
inline bool same(const Value& a, const Value& b);
inline uint64_t hash_of(const Value& v);
inline void repr_to(U32& out, const Value& v);
inline Value str_value(const Value& v);
inline std::string repr_utf8(const Value& v);
inline Value get_iter(const Value& v);
inline bool iter_next(const Value& iter, Value& out);
inline Value call_value(const Value& f, Value* args, int n, const TupleObj* kw_names = nullptr);
inline const char* tp_name(int type) { return type == T_deque ? "collections.deque" : type == T_defaultdict ? "collections.defaultdict" : kTypes[type].name; }
inline const char* type_name(const Value& v) { return tp_name(type_of(v)); }
inline std::string quoted(const Value& v) {
    return "'" + (v.tag == Tag::Instance ? std::string(v.as<InstanceObj>()->cls.as<ClassObj>()->name) : std::string(type_name(v))) + "'";
}

inline Value make_list(Vals&& v = {}) {
    auto* o = new ListObj;
    Value out(o);
    o->v = std::move(v);
    return out;
}
inline Value make_tuple(Vals&& v = {}) {
    auto* o = new TupleObj;
    Value out(o);
    o->v = std::move(v);
    return out;
}
inline Value pair(Value a, Value b) {
    Vals v;
    v.reserve(2);
    v.push_back(std::move(a));
    v.push_back(std::move(b));
    return make_tuple(std::move(v));
}
inline Value make_set(bool frozen = false) {
    auto* o = new SetObj;
    o->frozen = frozen;
    return Value(o);
}
inline Value make_dict(DictKind kind = DictKind::Plain) {
    auto* o = new DictObj;
    o->kind = kind;
    return Value(o);
}

inline int32_t Table::find(const Value& key, uint64_t hash, std::size_t* slot) const {
    if (index.empty()) return kFree;
    const std::size_t mask = index.size() - 1;
    std::size_t i = hash & mask, first_dummy = (std::size_t) -1;
    uint64_t perturb = hash;
    for (;;) {
        const int32_t at = index[i];
        if (at == kFree) {
            if (slot) *slot = first_dummy == (std::size_t) -1 ? i : first_dummy;
            return kFree;
        }
        if (at == kDummy) {
            if (first_dummy == (std::size_t) -1) first_dummy = i;
        } else {
            const Entry& e = entries[(std::size_t) at];
            if (e.hash == hash && same(e.key, key)) return at;
        }
        perturb >>= 5;
        i = (i * 5 + perturb + 1) & mask;
    }
}
inline void Table::rebuild(std::size_t want) {
    std::size_t size = 8;
    while (size * 2 < want * 3) size *= 2;
    rt->tick(entries.size() / 8 + 1);
    std::vector<Entry, Alloc<Entry>> kept;
    kept.reserve(size * 2 / 3 + 1);
    for (auto& e : entries)
        if (!e.key.empty()) kept.push_back(std::move(e));
    std::vector<int32_t, Alloc<int32_t>> fresh(size, kFree);
    for (std::size_t n = 0; n < kept.size(); ++n) {
        std::size_t i = kept[n].hash & (size - 1);
        uint64_t perturb = kept[n].hash;
        while (fresh[i] != kFree) {
            perturb >>= 5;
            i = (i * 5 + perturb + 1) & (size - 1);
        }
        fresh[i] = (int32_t) n;
    }
    entries.swap(kept);
    index.swap(fresh);
    ++rebuilds;
}
inline Value* Table::get(const Value& key) {
    if (index.empty()) {
        hash_of(key);
        return nullptr;
    }
    const int32_t at = find(key, hash_of(key));
    return at >= 0 ? &entries[(std::size_t) at].val : nullptr;
}
inline void Table::set(const Value& key, Value val) {
    const uint64_t hash = hash_of(key);
    if ((entries.size() + 1) * 3 >= index.size() * 2) rebuild((used + 1) * 2);
    std::size_t slot = 0;
    const int32_t at = find(key, hash, &slot);
    if (at >= 0) {
        entries[(std::size_t) at].val = std::move(val);
        return;
    }
    entries.push_back(Entry{hash, key, std::move(val)});
    index[slot] = (int32_t) entries.size() - 1;
    ++used;
}
inline bool Table::erase(const Value& key, Value* out) {
    const uint64_t hash = hash_of(key);
    const int32_t at = find(key, hash);
    if (at < 0) return false;
    Entry& e = entries[(std::size_t) at];
    if (out) *out = std::move(e.val);
    e.key.reset();
    e.val.reset();
    const std::size_t mask = index.size() - 1;
    std::size_t i = hash & mask;
    uint64_t perturb = hash;
    while (index[i] != at) {
        perturb >>= 5;
        i = (i * 5 + perturb + 1) & mask;
    }
    index[i] = kDummy;
    --used;
    return true;
}

// Sign and magnitude in 32-bit limbs. Values that fit stay in an int64_t, so one number has one form. Every operation
// charges its limb work before doing it: one step for about 32 limb products.
using Limbs = std::vector<uint32_t, Alloc<uint32_t>>;
struct BigObj : Obj {
    bool neg = false;
    Limbs mag;
    BigObj() : Obj(Tag::Big) {}
};
inline bool is_int(const Value& v) { return v.tag == Tag::Int || v.tag == Tag::Bool || v.tag == Tag::Big; }
inline bool is_number(const Value& v) { return is_int(v) || v.tag == Tag::Float; }
inline int64_t small_of(const Value& v) { return v.tag == Tag::Bool ? (v.b ? 1 : 0) : v.i; }
inline bool is_small(const Value& v) { return v.tag == Tag::Int || v.tag == Tag::Bool; }

// Any int read as sign and magnitude. Holds its own limbs, so it stays where it was made.
struct IntRef {
    bool neg;
    const uint32_t* p;
    std::size_t n;
    uint32_t small[2];
    explicit IntRef(const Value& v) {
        if (v.tag == Tag::Big) {
            const auto* b = v.as<BigObj>();
            neg = b->neg;
            p = b->mag.data();
            n = b->mag.size();
            return;
        }
        const int64_t i = small_of(v);
        neg = i < 0;
        const uint64_t m = neg ? 0 - (uint64_t) i : (uint64_t) i;
        small[0] = (uint32_t) m;
        small[1] = (uint32_t) (m >> 32);
        p = small;
        n = m == 0 ? 0 : small[1] ? 2 : 1;
    }
    IntRef(const IntRef&) = delete;
    IntRef& operator=(const IntRef&) = delete;
    std::size_t bits() const { return n ? (n - 1) * 32 + (std::size_t) std::bit_width(p[n - 1]) : 0; }
};
inline Value int_from(bool neg, Limbs&& m) {
    while (!m.empty() && m.back() == 0) m.pop_back();
    if (m.size() <= 2) {
        const uint64_t v = m.empty() ? 0 : m.size() == 1 ? m[0] : ((uint64_t) m[1] << 32 | m[0]);
        if (!neg && v <= (uint64_t) INT64_MAX) return Value::integer((int64_t) v);
        if (neg && v <= (uint64_t) INT64_MAX + 1) return Value::integer((int64_t) (0 - v));
    }
    auto* b = new BigObj;
    Value out(b);
    b->neg = neg;
    b->mag = std::move(m);
    return out;
}
inline int mag_cmp(const uint32_t* a, std::size_t n, const uint32_t* b, std::size_t m) {
    if (n != m) return n < m ? -1 : 1;
    for (std::size_t i = n; i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
inline int int_cmp(const Value& a, const Value& b) {
    if (is_small(a) && is_small(b)) {
        const int64_t x = small_of(a), y = small_of(b);
        return x < y ? -1 : x > y ? 1 : 0;
    }
    const IntRef x(a), y(b);
    if (x.neg != y.neg) return x.neg ? -1 : 1;
    const int c = mag_cmp(x.p, x.n, y.p, y.n);
    return x.neg ? -c : c;
}
inline Limbs mag_add(const uint32_t* a, std::size_t n, const uint32_t* b, std::size_t m) {
    if (n < m) {
        std::swap(a, b);
        std::swap(n, m);
    }
    rt->tick(8 + (n >> 6));
    Limbs out(n + 1);
    uint64_t carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
        carry += (uint64_t) a[i] + (i < m ? b[i] : 0);
        out[i] = (uint32_t) carry;
        carry >>= 32;
    }
    out[n] = (uint32_t) carry;
    return out;
}
inline Limbs mag_sub(const uint32_t* a, std::size_t n, const uint32_t* b, std::size_t m) {
    rt->tick(8 + (n >> 6));
    Limbs out(n);
    int64_t borrow = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int64_t t = (int64_t) a[i] - (i < m ? (int64_t) b[i] : 0) - borrow;
        out[i] = (uint32_t) t;
        borrow = t < 0 ? 1 : 0;
    }
    return out;
}
inline Limbs mag_mul(const uint32_t* a, std::size_t n, const uint32_t* b, std::size_t m) {
    rt->tick(8 + (((uint64_t) n * m) >> 5));
    Limbs out(n + m);
    for (std::size_t i = 0; i < n; ++i) {
        uint64_t carry = 0;
        const uint64_t x = a[i];
        for (std::size_t j = 0; j < m; ++j) {
            carry += x * b[j] + out[i + j];
            out[i + j] = (uint32_t) carry;
            carry >>= 32;
        }
        out[i + m] = (uint32_t) carry;
    }
    return out;
}
// Knuth's algorithm D. `m` > 0.
inline void mag_divmod(const uint32_t* a, std::size_t n, const uint32_t* b, std::size_t m, Limbs& q, Limbs& r) {
    if (mag_cmp(a, n, b, m) < 0) {
        q.clear();
        r.assign(a, a + n);
        return;
    }
    if (m == 1) {
        rt->tick(16 + (n >> 5));
        q.assign(n, 0);
        uint64_t rem = 0;
        for (std::size_t i = n; i-- > 0;) {
            rem = rem << 32 | a[i];
            q[i] = (uint32_t) (rem / b[0]);
            rem %= b[0];
        }
        r.assign(1, (uint32_t) rem);
        return;
    }
    rt->tick(16 + (((uint64_t) (n - m + 1) * m) >> 5));
    const int s = std::countl_zero(b[m - 1]);
    Limbs u(n + 1), v(m);
    uint64_t carry = 0;
    for (std::size_t i = 0; i < m; ++i) {
        carry |= (uint64_t) b[i] << s;
        v[i] = (uint32_t) carry;
        carry >>= 32;
    }
    carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
        carry |= (uint64_t) a[i] << s;
        u[i] = (uint32_t) carry;
        carry >>= 32;
    }
    u[n] = (uint32_t) carry;
    q.assign(n - m + 1, 0);
    constexpr uint64_t base = 1ull << 32;
    for (std::size_t j = n - m + 1; j-- > 0;) {
        const uint64_t top = ((uint64_t) u[j + m] << 32) | u[j + m - 1];
        uint64_t qhat = top / v[m - 1], rhat = top % v[m - 1];
        while (qhat >= base || qhat * v[m - 2] > ((rhat << 32) | u[j + m - 2])) {
            --qhat;
            rhat += v[m - 1];
            if (rhat >= base) break;
        }
        int64_t borrow = 0;
        uint64_t mul = 0;
        for (std::size_t i = 0; i < m; ++i) {
            const uint64_t p = qhat * v[i] + mul;
            mul = p >> 32;
            const int64_t t = (int64_t) u[i + j] - borrow - (int64_t) (p & 0xFFFFFFFFu);
            u[i + j] = (uint32_t) t;
            borrow = t < 0 ? 1 : 0;
        }
        const int64_t t = (int64_t) u[j + m] - borrow - (int64_t) mul;
        u[j + m] = (uint32_t) t;
        q[j] = (uint32_t) qhat;
        if (t < 0) {
            --q[j];
            uint64_t c = 0;
            for (std::size_t i = 0; i < m; ++i) {
                c += (uint64_t) u[i + j] + v[i];
                u[i + j] = (uint32_t) c;
                c >>= 32;
            }
            u[j + m] += (uint32_t) c;
        }
    }
    r.assign(m, 0);
    for (std::size_t i = 0; i < m; ++i) {
        const uint64_t w = ((uint64_t) u[i + 1] << 32 | u[i]) >> s;
        r[i] = (uint32_t) w;
    }
}
inline Value int_addsub(const Value& a, const Value& b, bool subtract) {
    const IntRef x(a), y(b);
    const bool yneg = y.n ? y.neg != subtract : false;
    if (!y.n || !x.n || x.neg == yneg) {
        if (!y.n) return a.tag == Tag::Bool ? Value::integer(small_of(a)) : a;
        if (!x.n) {
            Limbs copy(y.p, y.p + y.n);
            return int_from(yneg, std::move(copy));
        }
        return int_from(x.neg, mag_add(x.p, x.n, y.p, y.n));
    }
    const int c = mag_cmp(x.p, x.n, y.p, y.n);
    if (c == 0) return Value::integer(0);
    if (c > 0) return int_from(x.neg, mag_sub(x.p, x.n, y.p, y.n));
    return int_from(yneg, mag_sub(y.p, y.n, x.p, x.n));
}
inline Value int_add(const Value& a, const Value& b) {
    if (is_small(a) && is_small(b)) {
        const int64_t x = small_of(a), y = small_of(b), r = (int64_t) ((uint64_t) x + (uint64_t) y);
        if (((x ^ r) & (y ^ r)) >= 0) return Value::integer(r);
    }
    return int_addsub(a, b, false);
}
inline Value int_sub(const Value& a, const Value& b) {
    if (is_small(a) && is_small(b)) {
        const int64_t x = small_of(a), y = small_of(b), r = (int64_t) ((uint64_t) x - (uint64_t) y);
        if (((x ^ y) & (x ^ r)) >= 0) return Value::integer(r);
    }
    return int_addsub(a, b, true);
}
inline Value int_mul(const Value& a, const Value& b) {
    if (is_small(a) && is_small(b)) {
        const int64_t x = small_of(a), y = small_of(b);
        const bool neg = (x < 0) != (y < 0);
        const uint64_t ux = x < 0 ? 0 - (uint64_t) x : (uint64_t) x, uy = y < 0 ? 0 - (uint64_t) y : (uint64_t) y;
        const uint64_t a0 = (uint32_t) ux, a1 = ux >> 32, b0 = (uint32_t) uy, b1 = uy >> 32;
        const uint64_t ll = a0 * b0, lh = a0 * b1, hl = a1 * b0, hh = a1 * b1;
        const uint64_t mid = (ll >> 32) + (uint32_t) lh + (uint32_t) hl;
        const uint64_t high = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
        const uint64_t low = (mid << 32) | (uint32_t) ll;
        if (!high && low <= (uint64_t) INT64_MAX) return Value::integer(neg ? -(int64_t) low : (int64_t) low);
    }
    const IntRef x(a), y(b);
    if (!x.n || !y.n) return Value::integer(0);
    return int_from(x.neg != y.neg, mag_mul(x.p, x.n, y.p, y.n));
}
inline Value int_negate(const Value& a) {
    if (is_small(a) && small_of(a) != INT64_MIN) return Value::integer(-small_of(a));
    return int_sub(Value::integer(0), a);
}
inline Value int_abs(const Value& a) {
    const IntRef x(a);
    return x.neg ? int_negate(a) : a.tag == Tag::Bool ? Value::integer(small_of(a)) : a;
}
inline bool int_is_zero(const Value& a) { return is_small(a) ? small_of(a) == 0 : false; }
inline bool int_is_neg(const Value& a) { return is_small(a) ? small_of(a) < 0 : a.as<BigObj>()->neg; }
// Floored, as Python divides. `b` is not zero.
inline void int_divmod(const Value& a, const Value& b, Value* q, Value* r) {
    if (is_small(a) && is_small(b)) {
        const int64_t x = small_of(a), y = small_of(b);
        if (!(x == INT64_MIN && y == -1)) {
            int64_t d = x / y, m = x % y;
            if (m != 0 && (m < 0) != (y < 0)) {
                --d;
                m += y;
            }
            if (q) *q = Value::integer(d);
            if (r) *r = Value::integer(m);
            return;
        }
    }
    const IntRef x(a), y(b);
    Limbs lq, lr;
    mag_divmod(x.p, x.n, y.p, y.n, lq, lr);
    const bool neg = x.neg != y.neg;
    Value quotient = int_from(neg, std::move(lq));
    Value rest = int_from(x.neg, std::move(lr));
    if (!int_is_zero(rest) && neg) {
        quotient = int_sub(quotient, Value::integer(1));
        rest = int_add(rest, b);
    }
    if (q) *q = std::move(quotient);
    if (r) *r = std::move(rest);
}
inline Value int_floordiv(const Value& a, const Value& b) {
    Value q;
    int_divmod(a, b, &q, nullptr);
    return q;
}
inline Value int_mod(const Value& a, const Value& b) {
    Value r;
    int_divmod(a, b, nullptr, &r);
    return r;
}
inline std::size_t int_bits(const Value& a) {
    const IntRef x(a);
    return x.bits();
}
[[noreturn]] inline void too_big() { rt->charge(rt->limits.heap_bytes + 1); throw Stop{Status::MemoryLimit, "MemoryError"}; }
inline Value int_pow(const Value& base, uint64_t e) {
    const std::size_t bits = int_bits(base);
    if (bits > 1 && (double) (bits - 1) * (double) e > (double) rt->limits.heap_bytes * 8) too_big();
    Value r = Value::integer(1), b = base;
    while (e) {
        if (e & 1) r = int_mul(r, b);
        e >>= 1;
        if (e) b = int_mul(b, b);
    }
    return r;
}
inline Value int_gcd(Value a, Value b) {
    a = int_abs(a);
    b = int_abs(b);
    while (!int_is_zero(b)) {
        rt->tick();
        Value r = int_mod(a, b);
        a = std::move(b);
        b = std::move(r);
    }
    return a;
}
inline Value int_shl(const Value& a, uint64_t n) {
    const IntRef x(a);
    if (!x.n) return Value::integer(0);
    if (n / 8 > rt->limits.heap_bytes) too_big();
    const std::size_t words = (std::size_t) (n / 32);
    const int s = (int) (n % 32);
    rt->tick(x.n >> 6);
    Limbs out(x.n + words + 1);
    uint64_t carry = 0;
    for (std::size_t i = 0; i < x.n; ++i) {
        carry |= (uint64_t) x.p[i] << s;
        out[i + words] = (uint32_t) carry;
        carry >>= 32;
    }
    out[x.n + words] = (uint32_t) carry;
    return int_from(x.neg, std::move(out));
}
inline Value int_shr(const Value& a, uint64_t n) {
    if (int_is_neg(a)) return int_sub(Value::integer(-1), int_shr(int_sub(Value::integer(-1), a), n));
    const IntRef x(a);
    if (n / 32 >= x.n) return Value::integer(0);
    const std::size_t words = (std::size_t) (n / 32);
    const int s = (int) (n % 32);
    rt->tick(x.n >> 6);
    Limbs out(x.n - words);
    for (std::size_t i = words; i < x.n; ++i) {
        const uint64_t w = (uint64_t) x.p[i] | (i + 1 < x.n ? (uint64_t) x.p[i + 1] << 32 : 0);
        out[i - words] = (uint32_t) (w >> s);
    }
    return int_from(false, std::move(out));
}
inline Limbs twos_complement(const IntRef& x, std::size_t n) {
    Limbs out(n);
    std::copy(x.p, x.p + x.n, out.begin());
    if (x.neg) {
        uint64_t carry = 1;
        for (auto& w : out) {
            carry += (uint32_t) ~w;
            w = (uint32_t) carry;
            carry >>= 32;
        }
    }
    return out;
}
inline Value int_bitop(char op, const Value& a, const Value& b) {
    if (is_small(a) && is_small(b)) {
        const int64_t x = small_of(a), y = small_of(b);
        return Value::integer(op == '&' ? x & y : op == '|' ? x | y : x ^ y);
    }
    const IntRef x(a), y(b);
    const std::size_t n = std::max(x.n, y.n) + 1;
    Limbs p = twos_complement(x, n), q = twos_complement(y, n);
    rt->tick(n >> 6);
    for (std::size_t i = 0; i < n; ++i) p[i] = op == '&' ? p[i] & q[i] : op == '|' ? p[i] | q[i] : p[i] ^ q[i];
    const bool neg = p[n - 1] >> 31;
    if (neg) {
        uint64_t carry = 1;
        for (auto& w : p) {
            carry += (uint32_t) ~w;
            w = (uint32_t) carry;
            carry >>= 32;
        }
    }
    return int_from(neg, std::move(p));
}
inline std::string digit_limit_message(std::size_t digits, bool have_count) {
    std::string out = "Exceeds the limit (" + std::to_string(rt->limits.integer_digits) + " digits) for integer string conversion";
    if (have_count) out += ": value has " + std::to_string(digits) + " digits";
    return out + "; use sys.set_int_max_str_digits() to increase the limit";
}
// Decimal text, or ValueError past the digit limit as CPython 3.11 has it.
inline std::string int_decimal(const Value& v) {
    if (is_small(v)) return std::to_string(small_of(v));
    const IntRef x(v);
    const std::size_t limit = (std::size_t) rt->limits.integer_digits;
    if ((double) x.bits() * 0.30102999566398 > (double) limit + 1) raise(T_ValueError, digit_limit_message(0, false));
    rt->tick(((uint64_t) x.n * x.n) >> 4);
    Limbs work(x.p, x.p + x.n);
    std::string out;
    while (!work.empty()) {
        uint64_t rem = 0;
        for (std::size_t i = work.size(); i-- > 0;) {
            rem = rem << 32 | work[i];
            work[i] = (uint32_t) (rem / 1000000000u);
            rem %= 1000000000u;
        }
        while (!work.empty() && work.back() == 0) work.pop_back();
        for (int k = 0; k < 9 && (rem || !work.empty()); ++k) {
            out.push_back((char) ('0' + rem % 10));
            rem /= 10;
        }
    }
    if (out.size() > limit) raise(T_ValueError, digit_limit_message(0, false));
    if (x.neg) out.push_back('-');
    std::reverse(out.begin(), out.end());
    return out;
}
inline std::string int_base(const Value& v, int base) {
    const IntRef x(v);
    if (!x.n) return "0";
    const int shift = base == 16 ? 4 : base == 8 ? 3 : 1;
    rt->tick(x.n >> 4);
    std::string out;
    uint64_t buffer = 0;
    int have = 0;
    for (std::size_t i = 0; i < x.n; ++i) {
        buffer |= (uint64_t) x.p[i] << have;
        have += 32;
        for (; have >= shift; have -= shift, buffer >>= shift) out.push_back("0123456789abcdef"[buffer & (uint64_t) (base - 1)]);
    }
    if (buffer) out.push_back("0123456789abcdef"[buffer & (uint64_t) (base - 1)]);
    while (out.back() == '0') out.pop_back();
    std::reverse(out.begin(), out.end());
    return out;
}
// Digits already checked by the caller, `base` from 2 to 36.
inline Value int_parse(std::string_view digits, int base, bool neg) {
    const bool binary = base == 2 || base == 4 || base == 8 || base == 16 || base == 32;
    if (!binary && digits.size() > (std::size_t) rt->limits.integer_digits) raise(T_ValueError, digit_limit_message(digits.size(), true));
    rt->tick(((uint64_t) digits.size() * digits.size()) >> 8);
    Limbs out;
    out.reserve(digits.size() / 8 + 2);
    for (const char c : digits) {
        const int d = c <= '9' ? c - '0' : (c | 32) - 'a' + 10;
        uint64_t carry = (uint64_t) d;
        for (auto& w : out) {
            carry += (uint64_t) w * (uint64_t) base;
            w = (uint32_t) carry;
            carry >>= 32;
        }
        if (carry) out.push_back((uint32_t) carry);
    }
    return int_from(neg, std::move(out));
}
// m * 2^s, half to even. The low bit of `m` may stand for dropped bits; `m` is under 2^63.
inline double round_scaled(uint64_t m, int64_t s) {
    const int bits = std::bit_width(m);
    const int64_t drop = std::max<int64_t>(bits - 53, -1074 - s);
    if (drop > 0) {
        if (drop >= 64) return 0.0;
        const uint64_t low = m & ((1ull << drop) - 1), half = 1ull << (drop - 1);
        m >>= drop;
        if (low > half || (low == half && (m & 1))) ++m;
        s += drop;
    }
    if (s > 2000) return HUGE_VAL;
    return std::ldexp((double) m, (int) s);
}
inline double int_to_double(const Value& v) {
    if (is_small(v)) return (double) small_of(v);
    const IntRef x(v);
    const std::size_t bits = x.bits();
    if (bits > 1024) raise(T_OverflowError, "int too large to convert to float");
    const std::size_t shift = bits - 62;
    Value top = int_shr(int_abs(v), shift);
    uint64_t m = (uint64_t) top.i;
    for (std::size_t i = 0; i < shift / 32 && !(m & 1); ++i)
        if (x.p[i]) m |= 1;
    if (shift % 32 && (x.p[shift / 32] & ((1u << (shift % 32)) - 1))) m |= 1;
    const double r = round_scaled(m, (int64_t) shift);
    if (std::isinf(r)) raise(T_OverflowError, "int too large to convert to float");
    return x.neg ? -r : r;
}
inline Value int_from_double(double d) {
    if (std::fabs(d) < 9.0e18) return Value::integer((int64_t) d);
    int e = 0;
    const double m = std::frexp(std::fabs(d), &e);
    Value mantissa = Value::integer((int64_t) std::ldexp(m, 53));
    Value r = int_shl(mantissa, (uint64_t) (e - 53));
    return d < 0 ? int_negate(r) : r;
}
// True division, rounded correctly for any size of operand.
inline double int_true_div(const Value& a, const Value& b) {
    if (int_is_zero(b)) raise(T_ZeroDivisionError, "division by zero");
    constexpr int64_t exact = 1ll << 53;
    if (is_small(a) && is_small(b)) {
        const int64_t x = small_of(a), y = small_of(b);
        if (x >= -exact && x <= exact && y >= -exact && y <= exact) return (double) x / (double) y;
    }
    const bool neg = int_is_neg(a) != int_is_neg(b);
    if (int_is_zero(a)) return neg ? -0.0 : 0.0;
    const Value x = int_abs(a), y = int_abs(b);
    const int64_t gap = (int64_t) int_bits(x) - (int64_t) int_bits(y);
    if (gap > 1025) raise(T_OverflowError, "integer division result too large for a float");
    if (gap < -1100) return neg ? -0.0 : 0.0;
    const int64_t s = gap - 56;
    Value q, r;
    if (s >= 0) int_divmod(x, int_shl(y, (uint64_t) s), &q, &r);
    else int_divmod(int_shl(x, (uint64_t) -s), y, &q, &r);
    uint64_t m = (uint64_t) q.i;
    if (!int_is_zero(r)) m |= 1;
    const double out = round_scaled(m, s);
    if (std::isinf(out)) raise(T_OverflowError, "integer division result too large for a float");
    return neg ? -out : out;
}
// Exact: `2**53 + 1 == float(2**53)` is False. 2 for a NaN.
inline int int_cmp_double(const Value& v, double d) {
    if (std::isnan(d)) return 2;
    if (std::isinf(d)) return d > 0 ? -1 : 1;
    if (is_small(v)) {
        const int64_t i = small_of(v);
        if (i >= -(1ll << 53) && i <= (1ll << 53)) {
            const double x = (double) i;
            return x < d ? -1 : x > d ? 1 : 0;
        }
    }
    const double whole = std::floor(d);
    const int c = int_cmp(v, int_from_double(whole));
    if (c) return c;
    return d > whole ? -1 : 0;
}
inline constexpr uint64_t kHashModulus = (1ull << 61) - 1;
inline uint64_t hash_mix(int64_t h) { return h == -1 ? (uint64_t) -2 : (uint64_t) h; }
// CPython's numeric hash: equal numbers of any type hash the same.
inline uint64_t hash_int(const Value& v) {
    const IntRef x(v);
    rt->tick(x.n >> 4);
    uint64_t h = 0;
    for (std::size_t i = x.n; i-- > 0;) {
        h = ((h << 32) & kHashModulus) | (h >> 29);
        h += x.p[i];
        if (h >= kHashModulus) h -= kHashModulus;
    }
    return hash_mix(x.neg ? -(int64_t) h : (int64_t) h);
}
inline uint64_t hash_double(double v) {
    if (std::isinf(v)) return v > 0 ? 314159 : (uint64_t) -314159;
    if (std::isnan(v)) refuse("nan as a dict key or set member");
    int e = 0;
    double m = std::frexp(v, &e);
    int sign = 1;
    if (m < 0) {
        sign = -1;
        m = -m;
    }
    uint64_t x = 0;
    while (m != 0) {
        x = ((x << 28) & kHashModulus) | x >> (61 - 28);
        m *= 268435456.0;
        e -= 28;
        const uint64_t y = (uint64_t) m;
        m -= (double) y;
        x += y;
        if (x >= kHashModulus) x -= kHashModulus;
    }
    e = e >= 0 ? e % 61 : 61 - 1 - ((-1 - e) % 61);
    x = ((x << e) & kHashModulus) | x >> (61 - e);
    return hash_mix(sign < 0 ? -(int64_t) x : (int64_t) x);
}

// Decimal digits as David Gay's dtoa gives them: no zeros at either end, value = 0.digits * 10^point.
struct Digits {
    std::string digits;
    int point = 1;
};
inline Digits strip_digits(std::string all, int point) {
    std::size_t lead = 0;
    while (lead < all.size() && all[lead] == '0') ++lead;
    point -= (int) lead;
    all.erase(0, lead);
    while (!all.empty() && all.back() == '0') all.pop_back();
    return Digits{std::move(all), point};
}
inline Digits scientific_digits(const char* first, const char* last) {
    const char* e = std::find(first, last, 'e');
    int exponent = 0;
    std::from_chars(e + (e[1] == '+' ? 2 : 1), last, exponent);
    std::string all;
    for (const char* p = first; p != e; ++p)
        if (*p != '.') all.push_back(*p);
    return strip_digits(std::move(all), exponent + 1);
}
// `x` finite and not negative. mode 'r' shortest, 's' `n` significant digits, 'f' `n` digits after the point.
inline Digits float_digits(double x, char mode, int n) {
    if (x == 0) return Digits{"0", 1};
    std::string buffer((std::size_t) n + 400, '\0');
    char* const first = buffer.data();
    char* const end = first + buffer.size();
    if (mode == 'r') return scientific_digits(first, std::to_chars(first, end, x, std::chars_format::scientific).ptr);
    if (mode == 's') return scientific_digits(first, std::to_chars(first, end, x, std::chars_format::scientific, n - 1).ptr);
    char* const last = std::to_chars(first, end, x, std::chars_format::fixed, n).ptr;
    const char* dot = std::find((const char*) first, (const char*) last, '.');
    std::string all;
    for (const char* p = first; p != last; ++p)
        if (*p != '.') all.push_back(*p);
    Digits out = strip_digits(std::move(all), (int) (dot - first));
    if (out.digits.empty()) out.point = -n;
    return out;
}
// PyOS_double_to_string's layout, line for line: `x` finite and not negative, `type` one of r e f g.
inline std::string float_layout(double x, char type, int precision, bool alt, bool add_dot_0) {
    if (type == 'e') ++precision;
    if (type == 'g' && precision == 0) precision = 1;
    const Digits d = float_digits(x, type == 'r' ? 'r' : type == 'f' ? 'f' : 's', precision);
    const int length = (int) d.digits.size();
    int point = d.point, end = length, exponent = 0;
    bool use_exp = false;
    if (type == 'e') {
        use_exp = true;
        end = precision;
    } else if (type == 'f') {
        end = point + precision;
    } else if (type == 'g') {
        if (point <= -4 || point > precision - (add_dot_0 ? 1 : 0)) use_exp = true;
        if (alt) end = precision;
    } else if (point <= -4 || point > 16) {
        use_exp = true;
    }
    if (use_exp) {
        exponent = point - 1;
        point = 1;
    }
    const int start = point <= 0 ? point - 1 : 0;
    end = std::max(end, !use_exp && add_dot_0 ? point + 1 : point);
    std::string out;
    if (point <= 0) {
        out.append((std::size_t) (point - start), '0');
        out.push_back('.');
        out.append((std::size_t) (0 - point), '0');
    } else {
        out.append((std::size_t) (0 - start), '0');
    }
    if (0 < point && point <= length) {
        out.append(d.digits, 0, (std::size_t) point);
        out.push_back('.');
        out.append(d.digits, (std::size_t) point, std::string::npos);
    } else {
        out.append(d.digits);
    }
    if (length < point) {
        out.append((std::size_t) (point - length), '0');
        out.push_back('.');
        out.append((std::size_t) (end - point), '0');
    } else {
        out.append((std::size_t) (end - length), '0');
    }
    if (out.back() == '.' && !alt) out.pop_back();
    if (use_exp) {
        out.push_back('e');
        out.push_back(exponent < 0 ? '-' : '+');
        const int magnitude = exponent < 0 ? -exponent : exponent;
        if (magnitude < 10) out.push_back('0');
        out += std::to_string(magnitude);
    }
    return out;
}
inline std::string float_repr(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    return (std::signbit(x) ? "-" : "") + float_layout(std::fabs(x), 'r', 0, false, true);
}
// Python's rounding at `ndigits`: to the nearest decimal, half to even on the exact binary value.
inline double float_round(double x, int64_t ndigits) {
    if (!std::isfinite(x) || x == 0) return x;
    if (ndigits >= 0) {
        std::string buffer(800, '\0');
        char* const last = std::to_chars(buffer.data(), buffer.data() + buffer.size(), x, std::chars_format::fixed, (int) ndigits).ptr;
        double out = 0;
        std::from_chars(buffer.data(), last, out);
        return out;
    }
    std::string buffer(1600, '\0');
    char* const last = std::to_chars(buffer.data(), buffer.data() + buffer.size(), std::fabs(x), std::chars_format::fixed, 1100).ptr;
    const std::size_t whole = (std::size_t) (std::find(buffer.data(), last, '.') - buffer.data());
    const std::size_t drop = (std::size_t) -ndigits;
    if (drop > whole) return std::copysign(0.0, x);
    std::string kept(buffer.data(), whole - drop);
    const char next = buffer[whole - drop];
    bool sticky = false;
    for (const char* p = buffer.data() + (whole - drop) + 1; p != last; ++p)
        if (*p != '0' && *p != '.') sticky = true;
    const bool odd = !kept.empty() && ((kept.back() - '0') & 1);
    if (next > '5' || (next == '5' && (sticky || odd))) {
        std::size_t i = kept.size();
        for (; i > 0; --i) {
            if (kept[i - 1] != '9') {
                ++kept[i - 1];
                break;
            }
            kept[i - 1] = '0';
        }
        if (i == 0) kept.insert(kept.begin(), '1');
    }
    if (kept.empty()) kept = "0";
    kept.append(drop, '0');
    double out = 0;
    const auto done = std::from_chars(kept.data(), kept.data() + kept.size(), out);
    if (done.ec == std::errc::result_out_of_range) raise(T_OverflowError, "rounded value too large to represent");
    return std::copysign(out, x);
}
inline double parse_float_ascii(const std::string& text) {
    double out = 0;
    const auto done = std::from_chars(text.data(), text.data() + text.size(), out);
    if (done.ec == std::errc::result_out_of_range) return std::strtod(text.c_str(), nullptr);
    return out;
}

struct Spec {
    char32_t fill = ' ';
    char align = 0, sign = 0, group = 0, type = 0;
    bool alt = false, zero = false, coerce_zero = false, has_fill = false;
    int64_t width = 0, precision = -1;
};
inline Spec parse_spec(const U32& s, const char* type_name_for_error, bool text_type = false) {
    Spec sp;
    std::size_t i = 0;
    const std::size_t n = s.size();
    auto is_align = [](char32_t c) { return c == '<' || c == '>' || c == '=' || c == '^'; };
    if (n >= 2 && is_align(s[1])) {
        sp.fill = s[0];
        sp.has_fill = true;
        sp.align = (char) s[1];
        i = 2;
    } else if (n >= 1 && is_align(s[0])) {
        sp.align = (char) s[0];
        i = 1;
    }
    if (i < n && (s[i] == '+' || s[i] == '-' || s[i] == ' ')) sp.sign = (char) s[i++];
    if (i < n && s[i] == 'z') {
        sp.coerce_zero = true;
        ++i;
    }
    if (i < n && s[i] == '#') {
        sp.alt = true;
        ++i;
    }
    if (i < n && s[i] == '0') {
        sp.zero = true;
        ++i;
    }
    auto number = [&](int64_t& out) {
        const std::size_t begin = i;
        int64_t v = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9') {
            if (v > (INT64_MAX - 9) / 10) raise(T_ValueError, "Too many decimal digits in format string");
            v = v * 10 + (s[i++] - '0');
        }
        if (i != begin) out = v;
        return i != begin;
    };
    number(sp.width);
    if (i < n && (s[i] == ',' || s[i] == '_')) sp.group = (char) s[i++];
    if (i < n && sp.group && (s[i] == ',' || s[i] == '_') && s[i] != (char32_t) sp.group) raise(T_ValueError, "Cannot specify both ',' and '_'.");
    if (i < n && s[i] == '.') {
        ++i;
        if (!number(sp.precision)) raise(T_ValueError, "Format specifier missing precision");
    }
    if (n - i > 1) raise(T_ValueError, "Invalid format specifier '" + utf8(s) + "' for object of type '" + type_name_for_error + "'");
    if (i < n) {
        if (s[i] > 127) raise(T_ValueError, "Invalid format specifier '" + utf8(s) + "' for object of type '" + type_name_for_error + "'");
        sp.type = (char) s[i];
    }
    if (sp.group && sp.type) {
        const bool grouped = std::strchr("defgEG%F", sp.type) || (sp.group == '_' && std::strchr("boxX", sp.type));
        if (!grouped) raise(T_ValueError, std::string("Cannot specify '") + sp.group + "' with '" + sp.type + "'.");
    }
    if (sp.zero && !sp.has_fill) sp.fill = '0';
    if (sp.zero && !sp.align && !text_type) sp.align = '=';
    return sp;
}
inline U32 pad_to(U32&& body, int64_t width, char align, char32_t fill) {
    const int64_t have = (int64_t) body.size();
    if (have >= width) return std::move(body);
    const std::size_t missing = (std::size_t) (width - have);
    rt->tick(missing >> 6);
    const std::size_t left = align == '>' ? missing : align == '^' ? missing / 2 : 0;
    U32 out;
    out.reserve(body.size() + missing);
    out.append(left, fill);
    out += body;
    out.append(missing - left, fill);
    return out;
}
inline std::string with_groups(const std::string& digits, char sep, std::size_t every) {
    std::string out;
    for (std::size_t i = 0; i < digits.size(); ++i) {
        if (i && (digits.size() - i) % every == 0) out.push_back(sep);
        out.push_back(digits[i]);
    }
    return out;
}
// `head` is the sign and prefix, `whole` the digits that take grouping, `tail` what comes after them.
inline U32 lay_out_number(const Spec& sp, std::string head, std::string whole, const std::string& tail, std::size_t every = 3) {
    const char align = sp.align ? sp.align : '>';
    if (sp.group) {
        if (sp.fill == '0' && align == '=' && sp.width > 0) {
            const std::size_t target = (std::size_t) sp.width > head.size() + tail.size() ? (std::size_t) sp.width - head.size() - tail.size() : 0;
            rt->tick(target >> 6);
            while (whole.size() + (whole.size() - 1) / every < target) whole.insert(whole.begin(), '0');
        }
        whole = with_groups(whole, sp.group, every);
    }
    const std::size_t body = head.size() + whole.size() + tail.size();
    U32 out;
    if (align == '=' && (int64_t) body < sp.width) {
        const std::size_t missing = (std::size_t) sp.width - body;
        rt->tick(missing >> 6);
        out.reserve((std::size_t) sp.width);
        append(out, head);
        out.append(missing, sp.fill);
        append(out, whole);
        append(out, tail);
        return out;
    }
    out.reserve(body);
    append(out, head);
    append(out, whole);
    append(out, tail);
    return pad_to(std::move(out), sp.width, align, sp.fill);
}
inline std::string unknown_code(char code, const char* type) {
    return std::string("Unknown format code '") + code + "' for object of type '" + type + "'";
}
inline U32 format_float_spec(double x, const Spec& sp) {
    char type = sp.type;
    const bool upper = type == 'E' || type == 'F' || type == 'G';
    bool percent = false;
    if (sp.precision > 100000) refuse("a format precision past 100000");
    int precision = (int) sp.precision;
    bool add_dot_0 = false;
    char code = 0;
    switch (type) {
        case 0:
            add_dot_0 = true;
            code = precision < 0 ? 'r' : 'g';
            if (precision < 0) precision = 0;
            break;
        case 'n': refuse("the n format code");
        case '%':
            percent = true;
            code = 'f';
            x *= 100;
            break;
        case 'e': case 'E': code = 'e'; break;
        case 'f': case 'F': code = 'f'; break;
        case 'g': case 'G': code = 'g'; break;
        default: raise(T_ValueError, unknown_code(type, "float"));
    }
    if (precision < 0) precision = 6;
    const bool negative = std::signbit(x) && !std::isnan(x);
    std::string body;
    if (std::isnan(x)) body = "nan";
    else if (std::isinf(x)) body = "inf";
    else body = float_layout(std::fabs(x), code, precision, sp.alt, add_dot_0);
    if (upper)
        for (auto& ch : body) ch = (char) std::toupper((unsigned char) ch);
    if (percent) body.push_back('%');
    bool show_minus = negative;
    if (sp.coerce_zero && negative && body.find_first_of("123456789") == std::string::npos) show_minus = false;
    std::string head;
    if (show_minus) head = "-";
    else if (sp.sign == '+' || sp.sign == ' ') head = std::string(1, sp.sign);
    std::size_t split = 0;
    while (split < body.size() && std::isdigit((unsigned char) body[split])) ++split;
    return lay_out_number(sp, head, body.substr(0, split), body.substr(split));
}
inline U32 format_int_spec(const Value& v, const Spec& sp) {
    char type = sp.type;
    if (type == 0) type = 'd';
    if (type == 'e' || type == 'E' || type == 'f' || type == 'F' || type == 'g' || type == 'G' || type == '%') return format_float_spec(int_to_double(v), sp);
    if (type == 'n') refuse("the n format code");
    if (sp.precision >= 0) raise(T_ValueError, "Precision not allowed in integer format specifier");
    if (type == 'c') {
        if (sp.sign) raise(T_ValueError, "Sign not allowed with integer format specifier 'c'");
        if (!is_small(v) || small_of(v) > INT32_MAX || small_of(v) < INT32_MIN) raise(T_OverflowError, "Python int too large to convert to C long");
        const int64_t cp = small_of(v);
        if (cp < 0 || cp > 0x10FFFF) raise(T_OverflowError, "%c arg not in range(0x110000)");
        U32 one(1, (char32_t) cp);
        return pad_to(std::move(one), sp.width, sp.align ? sp.align : '<', sp.fill);
    }
    if (type != 'd' && type != 'b' && type != 'o' && type != 'x' && type != 'X') raise(T_ValueError, unknown_code(type, "int"));
    const bool negative = int_is_neg(v);
    std::string digits;
    if (type == 'd') {
        digits = int_decimal(int_abs(v));
    } else {
        digits = int_base(v, type == 'b' ? 2 : type == 'o' ? 8 : 16);
        if (type == 'X')
            for (auto& ch : digits) ch = (char) std::toupper((unsigned char) ch);
    }
    std::string head;
    if (negative) head = "-";
    else if (sp.sign == '+' || sp.sign == ' ') head = std::string(1, sp.sign);
    if (sp.alt && type != 'd') head += std::string("0") + type;
    return lay_out_number(sp, head, digits, "", type == 'd' ? 3 : 4);
}

inline int type_of(const Value& v) {
    switch (v.tag) {
        case Tag::None: return T_NoneType;
        case Tag::Bool: return T_bool;
        case Tag::Int: case Tag::Big: return T_int;
        case Tag::Float: return T_float;
        case Tag::Type: return T_type;
        case Tag::Native: return T_builtin_function_or_method;
        case Tag::Module: return T_module;
        case Tag::Ellipsis: return T_ellipsis;
        case Tag::Str: return T_str;
        case Tag::List: return T_list;
        case Tag::Tuple: return T_tuple;
        case Tag::Dict: {
            const DictKind kind = v.as<DictObj>()->kind;
            return kind == DictKind::Counter ? T_Counter : kind == DictKind::Default ? T_defaultdict : T_dict;
        }
        case Tag::Set: return v.as<SetObj>()->frozen ? T_frozenset : T_set;
        case Tag::Range: return T_range;
        case Tag::Slice: return T_slice;
        case Tag::Function: return T_function;
        case Tag::Bound: case Tag::Method: return T_method;
        case Tag::Class: return T_type;
        case Tag::Iter: return v.as<IterObj>()->type;
        case Tag::Exc: return v.as<ExcObj>()->type;
        case Tag::Cached: return T__lru_cache_wrapper;
        case Tag::Fraction: return T_Fraction;
        case Tag::Deque: return T_deque;
        case Tag::View: return v.as<ViewObj>()->type;
        case Tag::Stub: refuse("a typing object used as a value");
        default: return T_object;
    }
}
inline bool truth(const Value& v) {
    switch (v.tag) {
        case Tag::None: return false;
        case Tag::Bool: return v.b;
        case Tag::Int: return v.i != 0;
        case Tag::Float: return v.d != 0;
        case Tag::Str: return !str_of(v).empty();
        case Tag::List: return !v.as<ListObj>()->v.empty();
        case Tag::Tuple: return !v.as<TupleObj>()->v.empty();
        case Tag::Dict: return v.as<DictObj>()->t.used != 0;
        case Tag::Set: return v.as<SetObj>()->t.used != 0;
        case Tag::Range: return v.as<RangeObj>()->len != 0;
        case Tag::Deque: return v.as<DequeObj>()->count != 0;
        case Tag::View: return v.as<ViewObj>()->dict.as<DictObj>()->t.used != 0;
        case Tag::Fraction: return !int_is_zero(v.as<FractionObj>()->num);
        default: return true;
    }
}
inline const Vals& seq_of(const Value& v) { return v.tag == Tag::List ? v.as<ListObj>()->v : v.as<TupleObj>()->v; }

inline Value frac_make(Value n, Value d);
inline Value frac_from_double(double x);
inline double frac_to_double(const FractionObj* f) { return int_true_div(f->num, f->den); }
inline int frac_cmp(const Value& a, const Value& b);
inline bool is_rational(const Value& v) { return is_int(v) || v.tag == Tag::Fraction; }

inline bool seq_equal(const Vals& a, const Vals& b) {
    if (a.size() != b.size()) return false;
    Nest nest;
    rt->tick(a.size() >> 3);
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (!same(a[i], b[i])) return false;
    return true;
}
inline bool counter_count_equal(const DictObj* a, const DictObj* b) {
    auto covers = [](const DictObj* x, const DictObj* y) {
        for (const auto& e : x->t.entries) {
            if (e.key.empty()) continue;
            Value* other = const_cast<Table&>(y->t).get(e.key);
            if (!eq(e.val, other ? *other : Value::integer(0))) return false;
        }
        return true;
    };
    return covers(a, b) && covers(b, a);
}
inline bool table_subset(Table& a, Table& b) {
    rt->tick(a.used);
    for (const auto& e : a.entries)
        if (!e.key.empty() && !b.get(e.key)) return false;
    return true;
}
inline Table& keys_table(const Value& v) { return v.tag == Tag::Set ? v.as<SetObj>()->t : (v.tag == Tag::Dict ? v.as<DictObj>()->t : v.as<ViewObj>()->dict.as<DictObj>()->t); }
inline bool is_keys_like(const Value& v) { return v.tag == Tag::Set || (v.tag == Tag::View && v.as<ViewObj>()->type == T_dict_keys); }
inline Value items_to_set(const Value& view);
inline bool eq(const Value& a, const Value& b) {
    if (is_int(a) && is_int(b)) return int_cmp(a, b) == 0;
    if (a.tag == Tag::Float) {
        if (b.tag == Tag::Float) return a.d == b.d;
        if (is_int(b)) return int_cmp_double(b, a.d) == 0;
    }
    if (b.tag == Tag::Float && is_int(a)) return int_cmp_double(a, b.d) == 0;
    if (a.tag == Tag::Fraction || b.tag == Tag::Fraction) {
        if (a.tag == Tag::Float || b.tag == Tag::Float) {
            const double x = a.tag == Tag::Float ? a.d : b.d;
            if (!std::isfinite(x)) return false;
            return frac_cmp(a.tag == Tag::Float ? frac_from_double(x) : a, b.tag == Tag::Float ? frac_from_double(x) : b) == 0;
        }
        if (is_rational(a) && is_rational(b)) return frac_cmp(a, b) == 0;
        return false;
    }
    if (a.tag != b.tag) {
        if (is_keys_like(a) && is_keys_like(b)) {
            Nest nest;
            Table& x = keys_table(a);
            Table& y = keys_table(b);
            return x.used == y.used && table_subset(x, y);
        }
        return false;
    }
    switch (a.tag) {
        case Tag::Empty: case Tag::None: case Tag::Ellipsis: return true;
        case Tag::Type: case Tag::Module: return a.i == b.i;
        case Tag::Native: return a.native == b.native;
        case Tag::Str: return a.o == b.o || str_of(a) == str_of(b);
        case Tag::List: case Tag::Tuple: return seq_equal(seq_of(a), seq_of(b));
        case Tag::Dict: {
            auto* x = a.as<DictObj>();
            auto* y = b.as<DictObj>();
            if (x->kind == DictKind::Counter && y->kind == DictKind::Counter) return counter_count_equal(x, y);
            if (x->t.used != y->t.used) return false;
            Nest nest;
            rt->tick(x->t.used >> 2);
            for (const auto& e : x->t.entries) {
                if (e.key.empty()) continue;
                Value* other = y->t.get(e.key);
                if (!other || !same(e.val, *other)) return false;
            }
            return true;
        }
        case Tag::Set: {
            Nest nest;
            Table& x = a.as<SetObj>()->t;
            Table& y = b.as<SetObj>()->t;
            return x.used == y.used && table_subset(x, y);
        }
        case Tag::Range: {
            auto* x = a.as<RangeObj>();
            auto* y = b.as<RangeObj>();
            if (x->len != y->len) return false;
            if (x->len == 0) return true;
            if (x->start != y->start) return false;
            return x->len == 1 || x->step == y->step;
        }
        case Tag::Deque: {
            auto* x = a.as<DequeObj>();
            auto* y = b.as<DequeObj>();
            if (x->count != y->count) return false;
            Nest nest;
            for (std::size_t i = 0; i < x->count; ++i)
                if (!same(x->at(i), y->at(i))) return false;
            return true;
        }
        case Tag::View: {
            auto* x = a.as<ViewObj>();
            auto* y = b.as<ViewObj>();
            if (x->type == T_dict_values || y->type == T_dict_values) return a.o == b.o;
            if (x->type != y->type) return false;
            Nest nest;
            if (x->type == T_dict_keys) {
                Table& p = keys_table(a);
                Table& q = keys_table(b);
                return p.used == q.used && table_subset(p, q);
            }
            return eq(items_to_set(a), items_to_set(b));
        }
        case Tag::Slice: {
            auto* x = a.as<SliceObj>();
            auto* y = b.as<SliceObj>();
            return eq(x->start, y->start) && eq(x->stop, y->stop) && eq(x->step, y->step);
        }
        case Tag::Method: {
            auto* x = a.as<MethodObj>();
            auto* y = b.as<MethodObj>();
            return x->self.o == y->self.o && x->fn.o == y->fn.o;
        }
        default: return a.o == b.o;
    }
}
// `x is y or x == y`, what containers compare with. A float has no identity here, so a NaN cannot be told from another.
inline bool same(const Value& a, const Value& b) {
    if (a.tag >= Tag::Big && a.tag == b.tag && a.o == b.o) return true;
    if (a.tag == Tag::Float && b.tag == Tag::Float && std::isnan(a.d) && std::isnan(b.d)) refuse("identity of nan inside a container");
    return eq(a, b);
}
enum class Cmp : uint8_t { Lt, Le, Gt, Ge };
inline const char* cmp_text(Cmp op) { return op == Cmp::Lt ? "<" : op == Cmp::Le ? "<=" : op == Cmp::Gt ? ">" : ">="; }
inline bool cmp_result(Cmp op, int c) {
    switch (op) {
        case Cmp::Lt: return c < 0;
        case Cmp::Le: return c <= 0;
        case Cmp::Gt: return c > 0;
        default: return c >= 0;
    }
}
inline bool order(Cmp op, const Value& a, const Value& b);
inline bool order_seq(Cmp op, const Vals& a, const Vals& b) {
    Nest nest;
    rt->tick(std::min(a.size(), b.size()) >> 3);
    std::size_t i = 0;
    for (; i < a.size() && i < b.size(); ++i)
        if (!same(a[i], b[i])) return order(op, a[i], b[i]);
    return cmp_result(op, a.size() < b.size() ? -1 : a.size() > b.size() ? 1 : 0);
}
[[noreturn]] inline void no_order(Cmp op, const Value& a, const Value& b) {
    raise(T_TypeError, std::string("'") + cmp_text(op) + "' not supported between instances of " + quoted(a) + " and " + quoted(b));
}
inline bool order(Cmp op, const Value& a, const Value& b) {
    if (is_int(a) && is_int(b)) return cmp_result(op, int_cmp(a, b));
    if (a.tag == Tag::Float && b.tag == Tag::Float) {
        switch (op) {
            case Cmp::Lt: return a.d < b.d;
            case Cmp::Le: return a.d <= b.d;
            case Cmp::Gt: return a.d > b.d;
            default: return a.d >= b.d;
        }
    }
    if (a.tag == Tag::Float && is_int(b)) {
        const int c = int_cmp_double(b, a.d);
        return c != 2 && cmp_result(op, -c);
    }
    if (is_int(a) && b.tag == Tag::Float) {
        const int c = int_cmp_double(a, b.d);
        return c != 2 && cmp_result(op, c);
    }
    if (a.tag == Tag::Fraction || b.tag == Tag::Fraction) {
        if (a.tag == Tag::Float || b.tag == Tag::Float) {
            const double x = a.tag == Tag::Float ? a.d : b.d;
            if (std::isnan(x)) return false;
            if (std::isinf(x)) return cmp_result(op, (a.tag == Tag::Float) == (x > 0) ? 1 : -1);
            return cmp_result(op, frac_cmp(a.tag == Tag::Float ? frac_from_double(x) : a, b.tag == Tag::Float ? frac_from_double(x) : b));
        }
        if (is_rational(a) && is_rational(b)) return cmp_result(op, frac_cmp(a, b));
        no_order(op, a, b);
    }
    if (a.tag == Tag::Str && b.tag == Tag::Str) return cmp_result(op, str_of(a).compare(str_of(b)));
    if ((a.tag == Tag::List || a.tag == Tag::Tuple) && a.tag == b.tag) return order_seq(op, seq_of(a), seq_of(b));
    const bool a_keys = is_keys_like(a), b_keys = is_keys_like(b);
    if (a_keys && b_keys) {
        Nest nest;
        Table& x = keys_table(a);
        Table& y = keys_table(b);
        switch (op) {
            case Cmp::Le: return x.used <= y.used && table_subset(x, y);
            case Cmp::Lt: return x.used < y.used && table_subset(x, y);
            case Cmp::Ge: return x.used >= y.used && table_subset(y, x);
            default: return x.used > y.used && table_subset(y, x);
        }
    }
    if (a.tag == Tag::Deque && b.tag == Tag::Deque) {
        auto* x = a.as<DequeObj>();
        auto* y = b.as<DequeObj>();
        Nest nest;
        std::size_t i = 0;
        for (; i < x->count && i < y->count; ++i)
            if (!same(x->at(i), y->at(i))) return order(op, x->at(i), y->at(i));
        return cmp_result(op, x->count < y->count ? -1 : x->count > y->count ? 1 : 0);
    }
    if (a.tag == Tag::Dict && b.tag == Tag::Dict && a.as<DictObj>()->kind == DictKind::Counter && b.as<DictObj>()->kind == DictKind::Counter) {
        const Value& low = op == Cmp::Le || op == Cmp::Lt ? a : b;
        const Value& high = op == Cmp::Le || op == Cmp::Lt ? b : a;
        auto covers = [](const Value& x, const Value& y) {
            for (const auto& e : x.as<DictObj>()->t.entries) {
                if (e.key.empty()) continue;
                Value* other = y.as<DictObj>()->t.get(e.key);
                if (!order(Cmp::Le, e.val, other ? *other : Value::integer(0))) return false;
            }
            return true;
        };
        const bool within = covers(low, high) && covers(high, high);
        for (const auto& e : high.as<DictObj>()->t.entries) {
            if (e.key.empty()) continue;
            Value* other = low.as<DictObj>()->t.get(e.key);
            if (!order(Cmp::Le, other ? *other : Value::integer(0), e.val)) return false;
        }
        return within && (op == Cmp::Le || op == Cmp::Ge || !eq(a, b));
    }
    no_order(op, a, b);
}

inline uint64_t mix64(uint64_t h, uint64_t x) {
    h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h * 0xFF51AFD7ED558CCDull;
}
inline uint64_t hash_frac(const FractionObj* f);
inline uint64_t hash_of(const Value& v) {
    switch (v.tag) {
        case Tag::None: return 0x5D5A;
        case Tag::Bool: return v.b ? 1 : 0;
        case Tag::Int:
            if (v.i >= 0 && (uint64_t) v.i < kHashModulus) return (uint64_t) v.i;
            return hash_int(v);
        case Tag::Big: return hash_int(v);
        case Tag::Float: return hash_double(v.d);
        case Tag::Str: {
            auto* s = v.as<StrObj>();
            if (!s->hash) {
                rt->tick(s->s.size() >> 6);
                uint64_t h = 0xCBF29CE484222325ull;
                for (const char32_t c : s->s) h = (h ^ c) * 0x100000001B3ull;
                s->hash = h | 1;
            }
            return s->hash;
        }
        case Tag::Tuple: {
            // CPython's xxHash of the items: a tuple of numbers hashes to the same value there and here.
            constexpr uint64_t p1 = 11400714785074694791ull, p2 = 14029467366897019727ull, p5 = 2870177450012600261ull;
            Nest nest;
            const Vals& items = v.as<TupleObj>()->v;
            uint64_t acc = p5;
            for (const auto& item : items) {
                acc += hash_of(item) * p2;
                acc = (acc << 31) | (acc >> 33);
                acc *= p1;
            }
            acc += (uint64_t) items.size() ^ (p5 ^ 3527539ull);
            return acc == (uint64_t) -1 ? 1546275796 : acc;
        }
        case Tag::Set: {
            auto* s = v.as<SetObj>();
            if (!s->frozen) break;
            Nest nest;
            uint64_t h = 0x9D2C5680;
            for (const auto& e : s->t.entries)
                if (!e.key.empty()) h += e.hash * 0x100000001B3ull ^ (e.hash >> 29);
            return h;
        }
        case Tag::Range: {
            auto* r = v.as<RangeObj>();
            return mix64(mix64((uint64_t) r->len, r->len ? (uint64_t) r->start : 0), r->len > 1 ? (uint64_t) r->step : 0);
        }
        case Tag::Fraction: return hash_frac(v.as<FractionObj>());
        case Tag::Type: case Tag::Module: return mix64(77, (uint64_t) v.i);
        case Tag::Native: case Tag::Function: case Tag::Bound: case Tag::Iter: case Tag::Exc: case Tag::Cached: case Tag::Class: case Tag::Instance:
            return mix64(78, (uint64_t) v.i);
        case Tag::Method: return mix64(79, (uint64_t) v.as<MethodObj>()->self.i ^ (uint64_t) v.as<MethodObj>()->fn.i);
        case Tag::Ellipsis: return 0x5D5B;
        default: break;
    }
    raise(T_TypeError, "unhashable type: " + quoted(v));
}

inline bool printable(char32_t c) {
    if (c < 0x7F) return c >= 0x20;
    // Without Unicode's tables only blocks known to be printable from end to end pass; the caller refuses the rest.
    return (c >= 0xA1 && c <= 0xFF && c != 0xAD) || (c >= 0xE01 && c <= 0xE3A) || (c >= 0xE3F && c <= 0xE5B) || (c >= 0x4E00 && c <= 0x9FFF) ||
           (c >= 0x3041 && c <= 0x3096) || (c >= 0x30A1 && c <= 0x30FA) || (c >= 0xAC00 && c <= 0xD7A3) || (c >= 0x410 && c <= 0x44F) ||
           (c >= 0x1F300 && c <= 0x1F64F);
}
inline void repr_str(U32& out, const U32& s) {
    rt->tick(s.size() >> 6);
    const bool single = s.find(U'\'') != U32::npos, dbl = s.find(U'"') != U32::npos;
    const char32_t quote = single && !dbl ? U'"' : U'\'';
    out.push_back(quote);
    for (const char32_t c : s) {
        if (c == quote || c == U'\\') {
            out.push_back(U'\\');
            out.push_back(c);
        } else if (c == U'\n') append(out, "\\n");
        else if (c == U'\r') append(out, "\\r");
        else if (c == U'\t') append(out, "\\t");
        else if (printable(c)) out.push_back(c);
        else if (c < 0x100) {
            append(out, "\\x");
            out.push_back((unsigned char) "0123456789abcdef"[c >> 4]);
            out.push_back((unsigned char) "0123456789abcdef"[c & 15]);
        } else refuse("repr of a character outside the known printable blocks");
    }
    out.push_back(quote);
}
struct Busy {
    Obj* o;
    explicit Busy(Obj* x) : o(x) { o->busy = true; }
    ~Busy() { o->busy = false; }
    Busy(const Busy&) = delete;
    Busy& operator=(const Busy&) = delete;
};
inline void repr_items(U32& out, const Vals& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) append(out, ", ");
        repr_to(out, v[i]);
    }
}
inline void repr_dict_body(U32& out, DictObj* d) {
    out.push_back(U'{');
    bool first = true;
    for (std::size_t i = 0; i < d->t.entries.size(); ++i) {
        const Value key = d->t.entries[i].key, val = d->t.entries[i].val;
        if (key.empty()) continue;
        if (!first) append(out, ", ");
        first = false;
        repr_to(out, key);
        append(out, ": ");
        repr_to(out, val);
    }
    out.push_back(U'}');
}
inline Vals counter_sorted(DictObj* d);
inline void repr_table(U32& out, const Table& t, const char* before, const char* after) {
    append(out, before);
    bool first = true;
    for (const auto& e : t.entries) {
        if (e.key.empty()) continue;
        if (!first) append(out, ", ");
        first = false;
        repr_to(out, e.key);
    }
    append(out, after);
}
inline void repr_to(U32& out, const Value& v) {
    rt->tick();
    switch (v.tag) {
        case Tag::None: append(out, "None"); return;
        case Tag::Bool: append(out, v.b ? "True" : "False"); return;
        case Tag::Int: case Tag::Big: append(out, int_decimal(v)); return;
        case Tag::Float: append(out, float_repr(v.d)); return;
        case Tag::Ellipsis: append(out, "Ellipsis"); return;
        case Tag::Str: repr_str(out, str_of(v)); return;
        case Tag::Type: {
            if (v.i == T_iterator || v.i == T_method) refuse("the type of a C-level iterator or method");
            append(out, "<class '");
            if (v.i == T_Counter) append(out, "collections.");
            else if (v.i == T_Fraction) append(out, "fractions.");
            else if (v.i >= T_permutations && v.i <= T_islice) append(out, "itertools.");
            else if (v.i == T__lru_cache_wrapper) append(out, "functools.");
            append(out, tp_name((int) v.i));
            append(out, "'>");
            return;
        }
        case Tag::Native: append(out, "<built-in function "); append(out, v.native->name); append(out, ">"); return;
        default: break;
    }
    Obj* const o = v.tag >= Tag::Big ? v.o : nullptr;
    Nest nest;
    switch (v.tag) {
        case Tag::List: {
            if (o->busy) { append(out, "[...]"); return; }
            Busy busy(o);
            out.push_back(U'[');
            const Vals snapshot = seq_of(v);
            repr_items(out, snapshot);
            out.push_back(U']');
            return;
        }
        case Tag::Tuple: {
            if (o->busy) { append(out, "(...)"); return; }
            Busy busy(o);
            out.push_back(U'(');
            repr_items(out, seq_of(v));
            if (seq_of(v).size() == 1) out.push_back(U',');
            out.push_back(U')');
            return;
        }
        case Tag::Dict: {
            auto* d = v.as<DictObj>();
            if (d->kind == DictKind::Counter) {
                if (o->busy) { append(out, "..."); return; }
                Busy busy(o);
                if (!d->t.used) { append(out, "Counter()"); return; }
                append(out, "Counter({");
                const Vals items = counter_sorted(d);
                for (std::size_t i = 0; i < items.size(); ++i) {
                    if (i) append(out, ", ");
                    repr_to(out, seq_of(items[i])[0]);
                    append(out, ": ");
                    repr_to(out, seq_of(items[i])[1]);
                }
                append(out, "})");
                return;
            }
            if (d->kind == DictKind::Default) {
                if (o->busy) { append(out, "..."); return; }
                Busy busy(o);
                append(out, "defaultdict(");
                repr_to(out, d->factory);
                append(out, ", ");
                repr_dict_body(out, d);
                out.push_back(U')');
                return;
            }
            if (o->busy) { append(out, "{...}"); return; }
            Busy busy(o);
            repr_dict_body(out, d);
            return;
        }
        case Tag::Set: {
            auto* s = v.as<SetObj>();
            if (s->frozen) {
                if (!s->t.used) { append(out, "frozenset()"); return; }
                repr_table(out, s->t, "frozenset({", "})");
                return;
            }
            if (!s->t.used) { append(out, "set()"); return; }
            repr_table(out, s->t, "{", "}");
            return;
        }
        case Tag::Range: {
            auto* r = v.as<RangeObj>();
            append(out, "range(" + std::to_string(r->start) + ", " + std::to_string(r->stop));
            if (r->step != 1) append(out, ", " + std::to_string(r->step));
            out.push_back(U')');
            return;
        }
        case Tag::Slice: {
            auto* s = v.as<SliceObj>();
            append(out, "slice(");
            repr_to(out, s->start);
            append(out, ", ");
            repr_to(out, s->stop);
            append(out, ", ");
            repr_to(out, s->step);
            out.push_back(U')');
            return;
        }
        case Tag::Exc: {
            auto* e = v.as<ExcObj>();
            append(out, kTypes[e->type].name);
            out.push_back(U'(');
            repr_items(out, e->args);
            out.push_back(U')');
            return;
        }
        case Tag::Fraction: {
            auto* f = v.as<FractionObj>();
            append(out, "Fraction(" + int_decimal(f->num) + ", " + int_decimal(f->den) + ")");
            return;
        }
        case Tag::Deque: {
            auto* q = v.as<DequeObj>();
            if (o->busy) { append(out, "[...]"); return; }
            Busy busy(o);
            append(out, "deque([");
            for (std::size_t i = 0; i < q->count; ++i) {
                if (i) append(out, ", ");
                repr_to(out, q->at(i));
            }
            out.push_back(U']');
            if (q->maxlen >= 0) append(out, ", maxlen=" + std::to_string(q->maxlen));
            out.push_back(U')');
            return;
        }
        case Tag::View: {
            auto* w = v.as<ViewObj>();
            append(out, kTypes[w->type].name);
            append(out, "([");
            bool first = true;
            for (const auto& e : w->dict.as<DictObj>()->t.entries) {
                if (e.key.empty()) continue;
                if (!first) append(out, ", ");
                first = false;
                if (w->type == T_dict_keys) repr_to(out, e.key);
                else if (w->type == T_dict_values) repr_to(out, e.val);
                else {
                    out.push_back(U'(');
                    repr_to(out, e.key);
                    append(out, ", ");
                    repr_to(out, e.val);
                    out.push_back(U')');
                }
            }
            append(out, "])");
            return;
        }
        // These print a memory address in CPython, which no program can mean to use.
        default: refuse(std::string("repr of a ") + type_name(v));
    }
}
inline std::string repr_utf8(const Value& v) {
    U32 out;
    repr_to(out, v);
    return utf8(out);
}
inline Value str_value(const Value& v) {
    if (v.tag == Tag::Str) return v;
    U32 out;
    if (v.tag == Tag::Exc) {
        auto* e = v.as<ExcObj>();
        if (e->args.empty()) return text(U32{});
        if (e->args.size() == 1) {
            if (subtype(e->type, T_KeyError)) repr_to(out, e->args[0]);
            else return str_value(e->args[0]);
        } else {
            out.push_back(U'(');
            repr_items(out, e->args);
            out.push_back(U')');
        }
    } else if (v.tag == Tag::Fraction) {
        auto* f = v.as<FractionObj>();
        append(out, int_decimal(f->num));
        if (!eq(f->den, Value::integer(1))) append(out, "/" + int_decimal(f->den));
    } else {
        repr_to(out, v);
    }
    return text(std::move(out));
}

inline double to_float(const Value& v) {
    if (v.tag == Tag::Float) return v.d;
    if (is_int(v)) return int_to_double(v);
    return frac_to_double(v.as<FractionObj>());
}
inline double float_mod(double x, double y) {
    double mod = std::fmod(x, y);
    if (mod != 0) {
        if ((y < 0) != (mod < 0)) mod += y;
    } else {
        mod = std::copysign(0.0, y);
    }
    return mod;
}
inline void float_divmod(double x, double y, double& floordiv, double& mod) {
    mod = std::fmod(x, y);
    double div = (x - mod) / y;
    if (mod != 0) {
        if ((y < 0) != (mod < 0)) {
            mod += y;
            div -= 1.0;
        }
    } else {
        mod = std::copysign(0.0, y);
    }
    if (div != 0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, x / y);
    }
}
inline bool odd_integer(double x) { return std::fabs(std::fmod(x, 2.0)) == 1.0; }
inline double float_pow(double v, double w) {
    if (w == 0) return 1.0;
    if (std::isnan(v)) return v;
    if (std::isnan(w)) return v == 1.0 ? 1.0 : w;
    if (std::isinf(w)) {
        const double a = std::fabs(v);
        if (a == 1.0) return 1.0;
        if ((w > 0) == (a > 1.0)) return std::fabs(w);
        return 0.0;
    }
    if (std::isinf(v)) {
        const bool odd = odd_integer(w);
        if (w > 0) return odd ? v : std::fabs(v);
        return odd ? std::copysign(0.0, v) : 0.0;
    }
    if (v == 0) {
        if (w < 0) raise(T_ZeroDivisionError, "0.0 cannot be raised to a negative power");
        return odd_integer(w) ? v : 0.0;
    }
    bool negate = false;
    if (v < 0) {
        if (w != std::floor(w)) refuse("a negative number to a fractional power, which is complex");
        v = -v;
        negate = odd_integer(w);
    }
    if (v == 1.0) return negate ? -1.0 : 1.0;
    const double r = std::pow(v, w);
    if (std::isinf(r)) raise_value(T_OverflowError, pair(Value::integer(34), text("Result too large")));
    return negate ? -r : r;
}
enum class Bin : uint8_t { Add, Sub, Mul, Div, FloorDiv, Mod, Pow, LShift, RShift, And, Or, Xor, MatMul };
inline const char* bin_text(Bin op) {
    static const char* const names[] = {"+", "-", "*", "/", "//", "%", "** or pow()", "<<", ">>", "&", "|", "^", "@"};
    return names[(int) op];
}
[[noreturn]] inline void bad_operands(Bin op, const Value& a, const Value& b, bool inplace) {
    std::string name = bin_text(op);
    if (inplace) name = op == Bin::Pow ? "**=" : name + "=";
    raise(T_TypeError, "unsupported operand type(s) for " + name + ": " + quoted(a) + " and " + quoted(b));
}
inline std::size_t repeat_count(const Value& n) {
    if (n.tag == Tag::Big) raise(T_OverflowError, "cannot fit 'int' into an index-sized integer");
    const int64_t i = small_of(n);
    return i < 0 ? 0 : (std::size_t) i;
}
inline Value str_repeat(const U32& s, std::size_t n) {
    if (!n || s.empty()) return text(U32{});
    if ((double) n * (double) s.size() * sizeof(char32_t) > (double) rt->limits.heap_bytes) too_big();
    rt->tick((n * s.size()) >> 6);
    U32 out;
    out.reserve(n * s.size());
    for (std::size_t i = 0; i < n; ++i) out += s;
    return text(std::move(out));
}
inline Vals seq_repeat(const Vals& v, std::size_t n) {
    if (n && !v.empty() && (double) n * (double) v.size() * sizeof(Value) > (double) rt->limits.heap_bytes) too_big();
    rt->tick((n * v.size()) >> 2);
    Vals out;
    out.reserve(n * v.size());
    for (std::size_t i = 0; i < n; ++i) out.insert(out.end(), v.begin(), v.end());
    return out;
}
inline Value format_percent(const U32& format, const Value& args);
inline Value set_binary(Bin op, const Value& a, const Value& b, bool inplace);
inline Value frac_binary(Bin op, const Value& a, const Value& b);
inline Value counter_binary(Bin op, const Value& a, const Value& b);
inline Value counter_result(Bin op, const Value& a, const Value& b, bool inplace);
inline void list_extend(ListObj* list, const Value& items);
inline Value dict_copy(const Value& d);
inline void dict_update_from(DictObj* d, const Value& source);
inline Value deque_copy(const Value& d);
inline void deque_append(DequeObj* q, Value v);

inline bool is_iterable(const Value& v) {
    return v.tag == Tag::Iter || v.tag == Tag::List || v.tag == Tag::Tuple || v.tag == Tag::Str || v.tag == Tag::Dict || v.tag == Tag::Set ||
           v.tag == Tag::Range || v.tag == Tag::Deque || v.tag == Tag::View;
}
inline bool set_view(const Value& v) { return v.tag == Tag::View && v.as<ViewObj>()->type != T_dict_values; }
inline Value number_binary(Bin op, const Value& a, const Value& b, bool inplace) {
    if (is_int(a) && is_int(b)) {
        switch (op) {
            case Bin::Add: return int_add(a, b);
            case Bin::Sub: return int_sub(a, b);
            case Bin::Mul: return int_mul(a, b);
            case Bin::Div: return Value::real(int_true_div(a, b));
            case Bin::FloorDiv:
                if (int_is_zero(b)) raise(T_ZeroDivisionError, "integer division or modulo by zero");
                return int_floordiv(a, b);
            case Bin::Mod:
                if (int_is_zero(b)) raise(T_ZeroDivisionError, "integer modulo by zero");
                return int_mod(a, b);
            case Bin::Pow: {
                if (int_is_neg(b)) return Value::real(float_pow(int_to_double(a), int_to_double(b)));
                if (b.tag == Tag::Big) {
                    if (is_small(a) && small_of(a) >= -1 && small_of(a) <= 1) {
                        const int64_t x = small_of(a);
                        return Value::integer(x == -1 ? (b.as<BigObj>()->mag[0] & 1 ? -1 : 1) : x);
                    }
                    too_big();
                }
                return int_pow(a, (uint64_t) small_of(b));
            }
            case Bin::LShift:
                if (int_is_neg(b)) raise(T_ValueError, "negative shift count");
                if (b.tag == Tag::Big) {
                    if (int_is_zero(a)) return Value::integer(0);
                    too_big();
                }
                return int_shl(a, (uint64_t) small_of(b));
            case Bin::RShift:
                if (int_is_neg(b)) raise(T_ValueError, "negative shift count");
                if (b.tag == Tag::Big) return Value::integer(int_is_neg(a) ? -1 : 0);
                return int_shr(a, (uint64_t) small_of(b));
            case Bin::And: case Bin::Or: case Bin::Xor: {
                const char c = op == Bin::And ? '&' : op == Bin::Or ? '|' : '^';
                if (a.tag == Tag::Bool && b.tag == Tag::Bool) return Value::boolean(c == '&' ? a.b && b.b : c == '|' ? a.b || b.b : a.b != b.b);
                return int_bitop(c, a, b);
            }
            default: bad_operands(op, a, b, inplace);
        }
    }
    const double x = to_float(a), y = to_float(b);
    switch (op) {
        case Bin::Add: return Value::real(x + y);
        case Bin::Sub: return Value::real(x - y);
        case Bin::Mul: return Value::real(x * y);
        case Bin::Div:
            if (y == 0) raise(T_ZeroDivisionError, "float division by zero");
            return Value::real(x / y);
        case Bin::FloorDiv: {
            if (y == 0) raise(T_ZeroDivisionError, "float floor division by zero");
            double q = 0, m = 0;
            float_divmod(x, y, q, m);
            return Value::real(q);
        }
        case Bin::Mod:
            if (y == 0) raise(T_ZeroDivisionError, "float modulo");
            return Value::real(float_mod(x, y));
        case Bin::Pow: return Value::real(float_pow(x, y));
        default: bad_operands(op, a, b, inplace);
    }
}
inline Value binary(Bin op, const Value& a, const Value& b, bool inplace) {
    if (is_number(a) && is_number(b)) return number_binary(op, a, b, inplace);
    if ((a.tag == Tag::Fraction && (is_number(b) || b.tag == Tag::Fraction)) || (b.tag == Tag::Fraction && is_number(a))) return frac_binary(op, a, b);
    const bool a_seq = a.tag == Tag::Str || a.tag == Tag::List || a.tag == Tag::Tuple || a.tag == Tag::Deque;
    switch (op) {
        case Bin::Add:
            if (a.tag == Tag::Str) {
                if (b.tag != Tag::Str) raise(T_TypeError, std::string("can only concatenate str (not \"") + type_name(b) + "\") to str");
                const U32& x = str_of(a);
                const U32& y = str_of(b);
                rt->tick((x.size() + y.size()) >> 6);
                U32 out;
                out.reserve(x.size() + y.size());
                out += x;
                out += y;
                return text(std::move(out));
            }
            if (a.tag == Tag::List) {
                if (inplace) {
                    list_extend(a.as<ListObj>(), b);
                    return a;
                }
                if (b.tag != Tag::List) raise(T_TypeError, std::string("can only concatenate list (not \"") + type_name(b) + "\") to list");
            }
            if (a.tag == Tag::Tuple && b.tag != Tag::Tuple) raise(T_TypeError, std::string("can only concatenate tuple (not \"") + type_name(b) + "\") to tuple");
            if (a.tag == Tag::List || a.tag == Tag::Tuple) {
                const Vals& x = seq_of(a);
                const Vals& y = seq_of(b);
                rt->tick((x.size() + y.size()) >> 2);
                Vals out;
                out.reserve(x.size() + y.size());
                out.insert(out.end(), x.begin(), x.end());
                out.insert(out.end(), y.begin(), y.end());
                return a.tag == Tag::List ? make_list(std::move(out)) : make_tuple(std::move(out));
            }
            if (a.tag == Tag::Deque && b.tag == Tag::Deque) {
                Value out = inplace ? a : deque_copy(a);
                auto* src = b.as<DequeObj>();
                const std::size_t n = src->count;
                for (std::size_t i = 0; i < n; ++i) deque_append(out.as<DequeObj>(), Value(src->at(i)));
                return out;
            }
            if (a.tag == Tag::Dict && b.tag == Tag::Dict && a.as<DictObj>()->kind == DictKind::Counter && b.as<DictObj>()->kind == DictKind::Counter) return counter_result(op, a, b, inplace);
            break;
        case Bin::Mul: {
            const bool b_seq = b.tag == Tag::Str || b.tag == Tag::List || b.tag == Tag::Tuple || b.tag == Tag::Deque;
            if ((a_seq && !is_int(b)) || (b_seq && !is_int(a))) raise(T_TypeError, std::string("can't multiply sequence by non-int of type '") + type_name(a_seq ? b : a) + "'");
            if (a_seq || b_seq) {
                const Value& s = a_seq ? a : b;
                const std::size_t n = repeat_count(a_seq ? b : a);
                if (s.tag == Tag::Str) return str_repeat(str_of(s), n);
                if (s.tag == Tag::Deque) refuse("repeating a deque");
                if (inplace && s.tag == Tag::List && a_seq) {
                    Vals out = seq_repeat(seq_of(s), n);
                    s.as<ListObj>()->v.swap(out);
                    return s;
                }
                Vals out = seq_repeat(seq_of(s), n);
                return s.tag == Tag::List ? make_list(std::move(out)) : make_tuple(std::move(out));
            }
            break;
        }
        case Bin::Mod:
            if (a.tag == Tag::Str) return format_percent(str_of(a), b);
            break;
        case Bin::Or: case Bin::And: case Bin::Sub: case Bin::Xor:
            if (a.tag == Tag::Dict && b.tag == Tag::Dict) {
                auto* x = a.as<DictObj>();
                auto* y = b.as<DictObj>();
                if (x->kind == DictKind::Counter && y->kind == DictKind::Counter) return counter_result(op, a, b, inplace);
                if (op == Bin::Or) {
                    Value out = inplace ? a : dict_copy(a);
                    dict_update_from(out.as<DictObj>(), b);
                    return out;
                }
            }
            if (op == Bin::Or && inplace && a.tag == Tag::Dict) {
                dict_update_from(a.as<DictObj>(), b);
                return a;
            }
            if (a.tag == Tag::Set && b.tag == Tag::Set) return set_binary(op, a, b, inplace);
            // A keys or items view takes any iterable on either side, and what comes back is a set.
            if (set_view(a) || set_view(b)) {
                const Value left = a.tag == Tag::Set ? a : items_to_set(a), right = b.tag == Tag::Set ? b : items_to_set(b);
                return set_binary(op, left, right, false);
            }
            if (op == Bin::Or && (a.tag == Tag::Stub || a.tag == Tag::Type || a.tag == Tag::None) && (b.tag == Tag::Stub || b.tag == Tag::Type || b.tag == Tag::None) && (a.tag != Tag::None || b.tag != Tag::None)) return Value::of(Tag::Stub);
            break;
        default: break;
    }
    bad_operands(op, a, b, inplace);
}

inline std::size_t len_of(const Value& v) {
    switch (v.tag) {
        case Tag::Str: return str_of(v).size();
        case Tag::List: case Tag::Tuple: return seq_of(v).size();
        case Tag::Dict: return v.as<DictObj>()->t.used;
        case Tag::Set: return v.as<SetObj>()->t.used;
        case Tag::Range: return (std::size_t) v.as<RangeObj>()->len;
        case Tag::Deque: return v.as<DequeObj>()->count;
        case Tag::View: return v.as<ViewObj>()->dict.as<DictObj>()->t.used;
        default: raise(T_TypeError, "object of type " + quoted(v) + " has no len()");
    }
}
inline int64_t clamp_index(const Value& v) {
    if (is_small(v)) return small_of(v);
    return int_is_neg(v) ? INT64_MIN : INT64_MAX;
}
struct SliceRange {
    int64_t start, stop, step, count;
};
// PySlice_Unpack and PySlice_AdjustIndices.
inline SliceRange slice_range(const SliceObj* s, int64_t length) {
    auto check = [](const Value& v) {
        if (!v.is_none() && !is_int(v)) raise(T_TypeError, "slice indices must be integers or None or have an __index__ method");
    };
    check(s->start);
    check(s->stop);
    check(s->step);
    int64_t step = 1;
    if (!s->step.is_none()) {
        step = clamp_index(s->step);
        if (step == 0) raise(T_ValueError, "slice step cannot be zero");
        if (step == INT64_MIN) step = -INT64_MAX;
    }
    int64_t start = s->start.is_none() ? (step < 0 ? INT64_MAX : 0) : clamp_index(s->start);
    int64_t stop = s->stop.is_none() ? (step < 0 ? INT64_MIN : INT64_MAX) : clamp_index(s->stop);
    if (start < 0) {
        start = start == INT64_MIN ? (step < 0 ? -1 : 0) : start + length;
        if (start < 0) start = step < 0 ? -1 : 0;
    } else if (start >= length) {
        start = step < 0 ? length - 1 : length;
    }
    if (stop < 0) {
        stop = stop == INT64_MIN ? (step < 0 ? -1 : 0) : stop + length;
        if (stop < 0) stop = step < 0 ? -1 : 0;
    } else if (stop >= length) {
        stop = step < 0 ? length - 1 : length;
    }
    int64_t count = 0;
    if (step < 0) {
        if (stop < start) count = (start - stop - 1) / (-step) + 1;
    } else if (start < stop) {
        count = (stop - start - 1) / step + 1;
    }
    return SliceRange{start, stop, step, count};
}
inline int64_t seq_index(const Value& k, int64_t length, const char* what, bool store = false) {
    if (!is_small(k)) raise(T_IndexError, "cannot fit 'int' into an index-sized integer");
    int64_t i = small_of(k);
    if (i < 0 && i != INT64_MIN) i += length;
    if (i < 0 || i >= length) raise(T_IndexError, std::string(what) + (store ? " assignment index out of range" : " index out of range"));
    return i;
}
inline Value make_slice(Value a, Value b, Value c) {
    auto* s = new SliceObj;
    Value out(s);
    s->start = std::move(a);
    s->stop = std::move(b);
    s->step = std::move(c);
    return out;
}
inline Value dict_missing(DictObj* d, const Value& self, const Value& key) {
    if (d->kind == DictKind::Counter) return Value::integer(0);
    if (d->kind == DictKind::Default) {
        if (d->factory.is_none()) raise_value(T_KeyError, key);
        Value made = call_value(d->factory, nullptr, 0);
        self.as<DictObj>()->t.set(key, made);
        return made;
    }
    raise_value(T_KeyError, key);
}
inline Value subscript(const Value& c, const Value& k) {
    switch (c.tag) {
        case Tag::List: case Tag::Tuple: {
            const Vals& v = seq_of(c);
            const char* name = c.tag == Tag::List ? "list" : "tuple";
            if (is_int(k)) return v[(std::size_t) seq_index(k, (int64_t) v.size(), name)];
            if (k.tag != Tag::Slice) raise(T_TypeError, std::string(name) + " indices must be integers or slices, not " + type_name(k));
            const SliceRange r = slice_range(k.as<SliceObj>(), (int64_t) v.size());
            rt->tick((uint64_t) r.count >> 2);
            Vals out;
            out.reserve((std::size_t) r.count);
            int64_t at = r.start;
            for (int64_t i = 0; i < r.count; ++i, at += r.step) out.push_back(v[(std::size_t) at]);
            return c.tag == Tag::List ? make_list(std::move(out)) : make_tuple(std::move(out));
        }
        case Tag::Str: {
            const U32& s = str_of(c);
            if (is_int(k)) return text(U32(1, s[(std::size_t) seq_index(k, (int64_t) s.size(), "string")]));
            if (k.tag != Tag::Slice) raise(T_TypeError, std::string("string indices must be integers, not '") + type_name(k) + "'");
            const SliceRange r = slice_range(k.as<SliceObj>(), (int64_t) s.size());
            rt->tick((uint64_t) r.count >> 6);
            U32 out;
            out.reserve((std::size_t) r.count);
            int64_t at = r.start;
            for (int64_t i = 0; i < r.count; ++i, at += r.step) out.push_back(s[(std::size_t) at]);
            return text(std::move(out));
        }
        case Tag::Dict: {
            auto* d = c.as<DictObj>();
            if (Value* found = d->t.get(k)) return *found;
            return dict_missing(d, c, k);
        }
        case Tag::Range: {
            auto* r = c.as<RangeObj>();
            if (is_int(k)) {
                const int64_t i = seq_index(k, r->len, "range object");
                return Value::integer(r->start + i * r->step);
            }
            if (k.tag != Tag::Slice) raise(T_TypeError, std::string("range indices must be integers or slices, not ") + type_name(k));
            const SliceRange s = slice_range(k.as<SliceObj>(), r->len);
            const int64_t first = r->start + s.start * r->step;
            return Value(new RangeObj(first, r->start + s.stop * r->step, r->step * s.step, s.count));
        }
        case Tag::Deque: {
            auto* q = c.as<DequeObj>();
            if (!is_int(k)) raise(T_TypeError, std::string("sequence index must be integer, not '") + type_name(k) + "'");
            return q->at((std::size_t) seq_index(k, (int64_t) q->count, "deque"));
        }
        case Tag::Type: case Tag::Stub: return Value::of(Tag::Stub);
        default: raise(T_TypeError, quoted(c) + " object is not subscriptable");
    }
}
inline void store_slice(ListObj* list, const SliceObj* slice, const Value& value) {
    Vals items;
    if (value.tag == Tag::List || value.tag == Tag::Tuple) items = seq_of(value);
    else {
        if (!is_iterable(value)) raise(T_TypeError, "can only assign an iterable");
        Value it = get_iter(value);
        Value x;
        while (iter_next(it, x)) {
            rt->tick();
            items.push_back(std::move(x));
        }
    }
    Vals& v = list->v;
    const SliceRange r = slice_range(slice, (int64_t) v.size());
    rt->tick((v.size() + items.size()) >> 4);
    if (r.step == 1) {
        const int64_t stop = std::max(r.start, r.stop);
        Vals out;
        out.reserve(v.size() + items.size());
        out.insert(out.end(), v.begin(), v.begin() + r.start);
        out.insert(out.end(), items.begin(), items.end());
        out.insert(out.end(), v.begin() + stop, v.end());
        v.swap(out);
        return;
    }
    if ((int64_t) items.size() != r.count)
        raise(T_ValueError, "attempt to assign sequence of size " + std::to_string(items.size()) + " to extended slice of size " + std::to_string(r.count));
    int64_t at = r.start;
    for (int64_t i = 0; i < r.count; ++i, at += r.step) v[(std::size_t) at] = items[(std::size_t) i];
}
inline void store_subscript(const Value& c, const Value& k, const Value& value) {
    switch (c.tag) {
        case Tag::List: {
            Vals& v = c.as<ListObj>()->v;
            if (is_int(k)) {
                v[(std::size_t) seq_index(k, (int64_t) v.size(), "list", true)] = value;
                return;
            }
            if (k.tag != Tag::Slice) raise(T_TypeError, std::string("list indices must be integers or slices, not ") + type_name(k));
            store_slice(c.as<ListObj>(), k.as<SliceObj>(), value);
            return;
        }
        case Tag::Dict: c.as<DictObj>()->t.set(k, value); return;
        case Tag::Deque: {
            auto* q = c.as<DequeObj>();
            if (!is_int(k)) raise(T_TypeError, std::string("sequence index must be integer, not '") + type_name(k) + "'");
            q->at((std::size_t) seq_index(k, (int64_t) q->count, "deque", true)) = value;
            return;
        }
        default: raise(T_TypeError, quoted(c) + " object does not support item assignment");
    }
}
inline void delete_subscript(const Value& c, const Value& k) {
    switch (c.tag) {
        case Tag::List: {
            Vals& v = c.as<ListObj>()->v;
            if (is_int(k)) {
                const int64_t i = seq_index(k, (int64_t) v.size(), "list", true);
                rt->tick(v.size() >> 4);
                v.erase(v.begin() + i);
                return;
            }
            if (k.tag != Tag::Slice) raise(T_TypeError, std::string("list indices must be integers or slices, not ") + type_name(k));
            const SliceRange r = slice_range(k.as<SliceObj>(), (int64_t) v.size());
            rt->tick(v.size() >> 4);
            std::vector<bool, Alloc<bool>> gone(v.size(), false);
            int64_t at = r.start;
            for (int64_t i = 0; i < r.count; ++i, at += r.step) gone[(std::size_t) at] = true;
            Vals out;
            out.reserve(v.size());
            for (std::size_t i = 0; i < v.size(); ++i)
                if (!gone[i]) out.push_back(std::move(v[i]));
            v.swap(out);
            return;
        }
        case Tag::Dict:
            if (!c.as<DictObj>()->t.erase(k)) raise_value(T_KeyError, k);
            return;
        case Tag::Deque: {
            auto* q = c.as<DequeObj>();
            if (!is_int(k)) raise(T_TypeError, std::string("sequence index must be integer, not '") + type_name(k) + "'");
            const std::size_t i = (std::size_t) seq_index(k, (int64_t) q->count, "deque", true);
            ++q->version;
            for (std::size_t n = i; n + 1 < q->count; ++n) q->at(n) = std::move(q->at(n + 1));
            q->at(q->count - 1).reset();
            --q->count;
            return;
        }
        default: raise(T_TypeError, quoted(c) + " object doesn't support item deletion");
    }
}

struct ListIter : IterObj {
    Value seq;
    std::size_t i = 0;
    explicit ListIter(Value s) : seq(std::move(s)) {}
    void clear() override { seq.reset(); }
    bool next(Value& out) override {
        const Vals& v = seq_of(seq);
        if (i >= v.size()) return false;
        out = v[i++];
        return true;
    }
};
struct ReverseListIter : IterObj {
    Value seq;
    int64_t i;
    ReverseListIter(Value s, int64_t at) : IterObj(T_reversed), seq(std::move(s)), i(at) {}
    void clear() override { seq.reset(); }
    bool next(Value& out) override {
        const Vals& v = seq_of(seq);
        if (i < 0 || i >= (int64_t) v.size()) return false;
        out = v[(std::size_t) i--];
        return true;
    }
};
struct StrIter : IterObj {
    Value s;
    std::size_t i = 0;
    int direction;
    explicit StrIter(Value x, int d = 1) : s(std::move(x)), direction(d) {}
    void clear() override { s.reset(); }
    bool next(Value& out) override {
        const U32& t = str_of(s);
        if (i >= t.size()) return false;
        out = text(U32(1, direction > 0 ? t[i] : t[t.size() - 1 - i]));
        ++i;
        return true;
    }
};
struct RangeIter : IterObj {
    int64_t at, step, left;
    RangeIter(int64_t a, int64_t s, int64_t n) : at(a), step(s), left(n) {}
    bool next(Value& out) override {
        if (!left) return false;
        out = Value::integer(at);
        at += step;
        --left;
        return true;
    }
};
struct TableIter : IterObj {
    Value owner;
    int kind;  // T_dict_keys, T_dict_values, T_dict_items, T_set
    std::size_t i = 0, size;
    uint64_t rebuilds;
    TableIter(Value o, int k) : owner(std::move(o)), kind(k) {
        const Table& t = table();
        size = t.used;
        rebuilds = t.rebuilds;
    }
    void clear() override { owner.reset(); }
    Table& table() const { return owner.tag == Tag::Set ? owner.as<SetObj>()->t : owner.as<DictObj>()->t; }
    bool next(Value& out) override {
        Table& t = table();
        if (t.used != size) raise(T_RuntimeError, kind == T_set ? "Set changed size during iteration" : "dictionary changed size during iteration");
        if (t.rebuilds != rebuilds) raise(T_RuntimeError, kind == T_set ? "Set changed size during iteration" : "dictionary keys changed during iteration");
        while (i < t.entries.size() && t.entries[i].key.empty()) ++i;
        if (i >= t.entries.size()) return false;
        const Entry& e = t.entries[i++];
        if (kind == T_dict_values) out = e.val;
        else if (kind == T_dict_items) out = pair(e.key, e.val);
        else out = e.key;
        return true;
    }
};
struct DequeIter : IterObj {
    Value q;
    std::size_t i = 0;
    uint64_t version;
    int direction;
    DequeIter(Value d, int dir) : q(std::move(d)), version(q.as<DequeObj>()->version), direction(dir) {}
    void clear() override { q.reset(); }
    bool next(Value& out) override {
        auto* d = q.as<DequeObj>();
        if (d->version != version) raise(T_RuntimeError, "deque mutated during iteration");
        if (i >= d->count) return false;
        out = d->at(direction > 0 ? i : d->count - 1 - i);
        ++i;
        return true;
    }
};
inline Value get_iter(const Value& v) {
    switch (v.tag) {
        case Tag::Iter: return v;
        case Tag::List: case Tag::Tuple: return Value(new ListIter(v));
        case Tag::Str: return Value(new StrIter(v));
        case Tag::Range: {
            auto* r = v.as<RangeObj>();
            return Value(new RangeIter(r->start, r->step, r->len));
        }
        case Tag::Dict: return Value(new TableIter(v, T_dict_keys));
        case Tag::Set: return Value(new TableIter(v, T_set));
        case Tag::Deque: return Value(new DequeIter(v, 1));
        case Tag::View: return Value(new TableIter(v.as<ViewObj>()->dict, v.as<ViewObj>()->type));
        default: raise(T_TypeError, quoted(v) + " object is not iterable");
    }
}
inline bool iter_next(const Value& iter, Value& out) { return iter.as<IterObj>()->next(out); }
// Each item, one step each. A list is read live, so a body that appends is seen as CPython sees it.
template <class F>
inline void each(const Value& iterable, F&& f) {
    Value it = get_iter(iterable), x;
    while (iter_next(it, x)) {
        rt->tick();
        f(x);
    }
}
inline Vals to_vals(const Value& iterable) {
    Vals out;
    if (iterable.tag == Tag::List || iterable.tag == Tag::Tuple) {
        out = seq_of(iterable);
        rt->tick(out.size() >> 2);
        return out;
    }
    each(iterable, [&](Value& x) { out.push_back(std::move(x)); });
    return out;
}
inline bool contains(const Value& c, const Value& x) {
    switch (c.tag) {
        case Tag::List: case Tag::Tuple: {
            const Vals& v = seq_of(c);
            for (std::size_t i = 0; i < v.size(); ++i) {
                rt->tick();
                if (same(v[i], x)) return true;
            }
            return false;
        }
        case Tag::Str: {
            if (x.tag != Tag::Str) raise(T_TypeError, std::string("'in <string>' requires string as left operand, not ") + type_name(x));
            const U32& hay = str_of(c);
            const U32& needle = str_of(x);
            if (needle.size() == 1) rt->tick(hay.size() >> 5);
            if (needle.size() == 1) return hay.find(needle[0]) != U32::npos;
            rt->tick((hay.size() * std::min<uint64_t>(needle.size(), 64)) >> 5);
            return hay.find(needle) != U32::npos;
        }
        case Tag::Dict: return c.as<DictObj>()->t.get(x) != nullptr;
        case Tag::Set: return c.as<SetObj>()->t.get(x) != nullptr;
        case Tag::Range: {
            auto* r = c.as<RangeObj>();
            if (!is_int(x)) {
                if (x.tag != Tag::Float && x.tag != Tag::Fraction) return false;
                if (x.tag == Tag::Float && (!std::isfinite(x.d) || x.d != std::floor(x.d))) return false;
                if (x.tag == Tag::Fraction) return false;
                const Value whole = int_from_double(x.d);
                if (!is_small(whole)) return false;
                return contains(c, whole);
            }
            if (!is_small(x) || !r->len) return false;
            const int64_t i = small_of(x);
            if (r->step > 0 ? (i < r->start || i >= r->stop) : (i > r->start || i <= r->stop)) return false;
            return (i - r->start) % r->step == 0;
        }
        case Tag::View:
            if (c.as<ViewObj>()->type == T_dict_keys) return c.as<ViewObj>()->dict.as<DictObj>()->t.get(x) != nullptr;
            if (c.as<ViewObj>()->type == T_dict_items) {
                if (x.tag != Tag::Tuple || seq_of(x).size() != 2) return false;
                Value* found = c.as<ViewObj>()->dict.as<DictObj>()->t.get(seq_of(x)[0]);
                return found && same(*found, seq_of(x)[1]);
            }
            break;
        case Tag::Iter: case Tag::Deque: break;
        default: raise(T_TypeError, "argument of type " + quoted(c) + " is not iterable");
    }
    Value it = get_iter(c), item;
    while (iter_next(it, item)) {
        rt->tick();
        if (same(item, x)) return true;
    }
    return false;
}

inline bool is_space(char32_t c) {
    return (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
// Case and class of a non-ASCII character need Unicode's tables, which are not here. Refused, never guessed.
inline void need_ascii(const U32& s, const char* name) {
    for (const char32_t c : s)
        if (c >= 128) refuse(std::string(name) + " on non-ASCII text");
}
inline bool ascii_upper(char32_t c) { return c >= 'A' && c <= 'Z'; }
inline bool ascii_lower(char32_t c) { return c >= 'a' && c <= 'z'; }
inline bool ascii_digit(char32_t c) { return c >= '0' && c <= '9'; }
inline bool ascii_alpha(char32_t c) { return ascii_upper(c) || ascii_lower(c); }
inline char32_t to_lower(char32_t c) { return ascii_upper(c) ? c + 32 : c; }
inline char32_t to_upper(char32_t c) { return ascii_lower(c) ? c - 32 : c; }
inline void only_keywords(const Call& c, std::initializer_list<const char*> allowed, const char* who) {
    for (int k = 0; k < c.nkw(); ++k) {
        bool ok = false;
        for (const char* name : allowed) ok = ok || equals(c.kw_names->v[(std::size_t) k], name);
        if (!ok) raise(T_TypeError, repr_utf8(c.kw_names->v[(std::size_t) k]) + " is an invalid keyword argument for " + who + "()");
    }
}
inline const U32& self_str(const Call& c, const char* name) {
    if (c.a[0].tag != Tag::Str) raise(T_TypeError, std::string("descriptor '") + name + "' for 'str' objects doesn't apply to a " + quoted(c.a[0]) + " object");
    return str_of(c.a[0]);
}
inline const U32& str_arg(const Value& v, const char* what) {
    if (v.tag != Tag::Str) raise(T_TypeError, std::string(what) + (*what ? " must" : "must") + " be str, not " + type_name(v));
    return str_of(v);
}
inline int64_t int_arg(const Value& v) {
    if (!is_int(v)) raise(T_TypeError, quoted(v) + " object cannot be interpreted as an integer");
    return clamp_index(v);
}
struct Bounds {
    int64_t start, stop;
};
inline Bounds bounds_arg(const Call& c, int at, int64_t length) {
    int64_t start = 0, stop = length;
    if (c.n > at && !c.a[at].is_none()) start = int_arg(c.a[at]);
    if (c.n > at + 1 && !c.a[at + 1].is_none()) stop = int_arg(c.a[at + 1]);
    if (stop > length) stop = length;
    else if (stop < 0) stop = std::max<int64_t>(stop + length, 0);
    if (start < 0) start = std::max<int64_t>(start + length, 0);
    return Bounds{start, stop};
}
inline int64_t find_in(const U32& hay, const U32& needle, int64_t start, int64_t stop) {
    const int64_t m = (int64_t) needle.size();
    uint64_t work = 0;
    for (int64_t i = start; i + m <= stop; ++i) {
        int64_t j = 0;
        while (j < m && hay[(std::size_t) (i + j)] == needle[(std::size_t) j]) ++j;
        work += (uint64_t) j + 1;
        if (j == m) {
            rt->tick(1 + (work >> 3));
            return i;
        }
        if (work > 4096) {
            rt->tick(work >> 3);
            work = 0;
        }
    }
    rt->tick(work >> 3);
    return -1;
}
inline int64_t rfind_in(const U32& hay, const U32& needle, int64_t start, int64_t stop) {
    const int64_t m = (int64_t) needle.size();
    uint64_t work = 0;
    for (int64_t i = stop - m; i >= start; --i) {
        int64_t j = 0;
        while (j < m && hay[(std::size_t) (i + j)] == needle[(std::size_t) j]) ++j;
        work += (uint64_t) j + 1;
        if (j == m) {
            rt->tick(1 + (work >> 3));
            return i;
        }
        if (work > 4096) {
            rt->tick(work >> 3);
            work = 0;
        }
    }
    rt->tick(work >> 3);
    return -1;
}
inline Value s_find(Call& c) {
    const U32& s = self_str(c, "find");
    const U32& sub = str_arg(c.a[1], "");
    const Bounds b = bounds_arg(c, 2, (int64_t) s.size());
    return Value::integer(find_in(s, sub, b.start, b.stop));
}
inline Value s_rfind(Call& c) {
    const U32& s = self_str(c, "rfind");
    const U32& sub = str_arg(c.a[1], "");
    const Bounds b = bounds_arg(c, 2, (int64_t) s.size());
    return Value::integer(rfind_in(s, sub, b.start, b.stop));
}
inline Value s_index(Call& c) {
    const Value r = s_find(c);
    if (r.i < 0) raise(T_ValueError, "substring not found");
    return r;
}
inline Value s_rindex(Call& c) {
    const Value r = s_rfind(c);
    if (r.i < 0) raise(T_ValueError, "substring not found");
    return r;
}
inline Value s_count(Call& c) {
    const U32& s = self_str(c, "count");
    const U32& sub = str_arg(c.a[1], "");
    const Bounds b = bounds_arg(c, 2, (int64_t) s.size());
    if (b.start > b.stop) return Value::integer(0);
    if (sub.empty()) return Value::integer(b.stop - b.start + 1);
    int64_t n = 0;
    for (int64_t at = b.start; (at = find_in(s, sub, at, b.stop)) >= 0; at += (int64_t) sub.size()) ++n;
    return Value::integer(n);
}
inline bool edge_match(const U32& s, const U32& part, Bounds b, bool at_end) {
    if (b.start > b.stop) return false;
    const int64_t m = (int64_t) part.size();
    if (b.stop - b.start < m) return false;
    const int64_t at = at_end ? b.stop - m : b.start;
    return s.compare((std::size_t) at, (std::size_t) m, part) == 0;
}
inline Value edge_test(Call& c, const char* name, bool at_end) {
    const U32& s = self_str(c, name);
    const Bounds b = bounds_arg(c, 2, (int64_t) s.size());
    const Value& what = c.a[1];
    if (what.tag == Tag::Tuple) {
        for (const auto& item : seq_of(what)) {
            if (item.tag != Tag::Str) raise(T_TypeError, std::string("tuple for ") + name + " must only contain str, not " + type_name(item));
            if (edge_match(s, str_of(item), b, at_end)) return Value::boolean(true);
        }
        return Value::boolean(false);
    }
    if (what.tag != Tag::Str) raise(T_TypeError, std::string(name) + " first arg must be str or a tuple of str, not " + type_name(what));
    return Value::boolean(edge_match(s, str_of(what), b, at_end));
}
inline Value s_startswith(Call& c) { return edge_test(c, "startswith", false); }
inline Value s_endswith(Call& c) { return edge_test(c, "endswith", true); }
inline Value slice_text(const U32& s, std::size_t a, std::size_t b) {
    rt->tick((b - a) >> 6);
    return text(U32(s, a, b - a));
}
inline Value s_split(Call& c) {
    const U32& s = self_str(c, "split");
    only_keywords(c, {"sep", "maxsplit"}, "split");
    Value* kw_sep = c.keyword("sep");
    Value* kw_max = c.keyword("maxsplit");
    const Value sep = kw_sep ? *kw_sep : c.n > 1 ? c.a[1] : Value::none();
    int64_t left = kw_max ? int_arg(*kw_max) : c.n > 2 ? int_arg(c.a[2]) : -1;
    if (left < 0) left = INT64_MAX;
    Vals out;
    if (sep.is_none()) {
        std::size_t i = 0;
        for (;;) {
            while (i < s.size() && is_space(s[i])) ++i;
            if (i >= s.size()) break;
            std::size_t j = i;
            if (left == 0) {
                out.push_back(slice_text(s, i, s.size()));
                break;
            }
            while (j < s.size() && !is_space(s[j])) ++j;
            out.push_back(slice_text(s, i, j));
            --left;
            i = j;
        }
        return make_list(std::move(out));
    }
    if (sep.tag != Tag::Str) raise(T_TypeError, std::string("must be str or None, not ") + type_name(sep));
    const U32& d = str_of(sep);
    if (d.empty()) raise(T_ValueError, "empty separator");
    int64_t at = 0;
    while (left-- > 0) {
        const int64_t hit = find_in(s, d, at, (int64_t) s.size());
        if (hit < 0) break;
        out.push_back(slice_text(s, (std::size_t) at, (std::size_t) hit));
        at = hit + (int64_t) d.size();
    }
    out.push_back(slice_text(s, (std::size_t) at, s.size()));
    return make_list(std::move(out));
}
inline Value s_rsplit(Call& c) {
    const U32& s = self_str(c, "rsplit");
    only_keywords(c, {"sep", "maxsplit"}, "rsplit");
    Value* kw_sep = c.keyword("sep");
    Value* kw_max = c.keyword("maxsplit");
    const Value sep = kw_sep ? *kw_sep : c.n > 1 ? c.a[1] : Value::none();
    int64_t left = kw_max ? int_arg(*kw_max) : c.n > 2 ? int_arg(c.a[2]) : -1;
    if (left < 0) left = INT64_MAX;
    Vals out;
    if (sep.is_none()) {
        std::size_t i = s.size();
        for (;;) {
            while (i > 0 && is_space(s[i - 1])) --i;
            if (i == 0) break;
            std::size_t j = i;
            if (left == 0) {
                out.push_back(slice_text(s, 0, i));
                break;
            }
            while (j > 0 && !is_space(s[j - 1])) --j;
            out.push_back(slice_text(s, j, i));
            --left;
            i = j;
        }
    } else {
        if (sep.tag != Tag::Str) raise(T_TypeError, std::string("must be str or None, not ") + type_name(sep));
        const U32& d = str_of(sep);
        if (d.empty()) raise(T_ValueError, "empty separator");
        int64_t end = (int64_t) s.size();
        while (left-- > 0) {
            const int64_t hit = rfind_in(s, d, 0, end);
            if (hit < 0) break;
            out.push_back(slice_text(s, (std::size_t) hit + d.size(), (std::size_t) end));
            end = hit;
        }
        out.push_back(slice_text(s, 0, (std::size_t) end));
    }
    std::reverse(out.begin(), out.end());
    return make_list(std::move(out));
}
inline Value s_splitlines(Call& c) {
    const U32& s = self_str(c, "splitlines");
    only_keywords(c, {"keepends"}, "splitlines");
    Value* kw = c.keyword("keepends");
    const bool keep = kw ? truth(*kw) : c.n > 1 ? truth(c.a[1]) : false;
    Vals out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = i;
        auto line_end = [](char32_t x) { return x == 10 || x == 13 || x == 11 || x == 12 || (x >= 0x1C && x <= 0x1E) || x == 0x85 || x == 0x2028 || x == 0x2029; };
        while (j < s.size() && !line_end(s[j])) ++j;
        std::size_t next = j;
        if (j < s.size()) next = s[j] == 13 && j + 1 < s.size() && s[j + 1] == 10 ? j + 2 : j + 1;
        out.push_back(slice_text(s, i, keep ? next : j));
        i = next;
    }
    return make_list(std::move(out));
}
inline Value s_join(Call& c) {
    const U32& sep = self_str(c, "join");
    if (!is_iterable(c.a[1])) raise(T_TypeError, "can only join an iterable");
    const Vals items = to_vals(c.a[1]);
    std::size_t total = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (items[i].tag != Tag::Str) raise(T_TypeError, "sequence item " + std::to_string(i) + ": expected str instance, " + type_name(items[i]) + " found");
        total += str_of(items[i]).size() + sep.size();
    }
    rt->tick(total >> 6);
    U32 out;
    out.reserve(total);
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out += sep;
        out += str_of(items[i]);
    }
    return text(std::move(out));
}
inline Value s_replace(Call& c) {
    const U32& s = self_str(c, "replace");
    const U32& from = str_arg(c.a[1], "replace() argument 1");
    const U32& to = str_arg(c.a[2], "replace() argument 2");
    int64_t left = c.n > 3 ? int_arg(c.a[3]) : -1;
    if (left < 0) left = INT64_MAX;
    U32 out;
    if (from.empty()) {
        if ((double) s.size() * (double) to.size() * 4 > (double) rt->limits.heap_bytes) too_big();
        for (std::size_t i = 0; i <= s.size(); ++i) {
            if (left > 0) {
                out += to;
                --left;
            }
            if (i < s.size()) out.push_back(s[i]);
        }
        rt->tick(out.size() >> 6);
        return text(std::move(out));
    }
    int64_t at = 0;
    while (left-- > 0) {
        const int64_t hit = find_in(s, from, at, (int64_t) s.size());
        if (hit < 0) break;
        rt->tick(2 + ((to.size() + (uint64_t) (hit - at)) >> 4));
        out.append(s, (std::size_t) at, (std::size_t) (hit - at));
        out += to;
        at = hit + (int64_t) from.size();
    }
    rt->tick(out.size() >> 6);
    out.append(s, (std::size_t) at, std::string::npos);
    return text(std::move(out));
}
inline Value map_case(Call& c, const char* name, char32_t (*f)(char32_t)) {
    const U32& s = self_str(c, name + 4);
    need_ascii(s, name);
    rt->tick(s.size() >> 6);
    U32 out = s;
    for (auto& x : out) x = f(x);
    return text(std::move(out));
}
inline Value s_upper(Call& c) { return map_case(c, "str.upper", to_upper); }
inline Value s_lower(Call& c) { return map_case(c, "str.lower", to_lower); }
inline Value s_casefold(Call& c) { return map_case(c, "str.casefold", to_lower); }
inline Value s_swapcase(Call& c) { return map_case(c, "str.swapcase", [](char32_t x) -> char32_t { return ascii_upper(x) ? to_lower(x) : to_upper(x); }); }
inline Value s_capitalize(Call& c) {
    const U32& s = self_str(c, "capitalize");
    need_ascii(s, "str.capitalize");
    U32 out = s;
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = i ? to_lower(out[i]) : to_upper(out[i]);
    return text(std::move(out));
}
inline Value s_title(Call& c) {
    const U32& s = self_str(c, "title");
    need_ascii(s, "str.title");
    U32 out = s;
    bool after_letter = false;
    for (auto& x : out) {
        x = after_letter ? to_lower(x) : to_upper(x);
        after_letter = ascii_alpha(x);
    }
    return text(std::move(out));
}
template <class F>
inline Value class_test(const Value& self, const char* name, F f) {
    const U32& s = str_of(self);
    need_ascii(s, name);
    if (s.empty()) return Value::boolean(false);
    for (const char32_t x : s)
        if (!f(x)) return Value::boolean(false);
    return Value::boolean(true);
}
inline Value s_isdigit(Call& c) { self_str(c, "isdigit"); return class_test(c.a[0], "str.isdigit", ascii_digit); }
inline Value s_isdecimal(Call& c) { self_str(c, "isdecimal"); return class_test(c.a[0], "str.isdecimal", ascii_digit); }
inline Value s_isnumeric(Call& c) { self_str(c, "isnumeric"); return class_test(c.a[0], "str.isnumeric", ascii_digit); }
inline Value s_isalpha(Call& c) { self_str(c, "isalpha"); return class_test(c.a[0], "str.isalpha", ascii_alpha); }
inline Value s_isalnum(Call& c) { self_str(c, "isalnum"); return class_test(c.a[0], "str.isalnum", [](char32_t x) { return ascii_alpha(x) || ascii_digit(x); }); }
inline Value s_isspace(Call& c) {
    const U32& s = self_str(c, "isspace");
    if (s.empty()) return Value::boolean(false);
    return Value::boolean(std::all_of(s.begin(), s.end(), is_space));
}
inline Value s_isascii(Call& c) {
    const U32& s = self_str(c, "isascii");
    return Value::boolean(std::all_of(s.begin(), s.end(), [](char32_t x) { return x < 128; }));
}
inline Value s_isprintable(Call& c) {
    const U32& s = self_str(c, "isprintable");
    need_ascii(s, "str.isprintable");
    return Value::boolean(std::all_of(s.begin(), s.end(), printable));
}
inline Value s_isidentifier(Call& c) {
    const U32& s = self_str(c, "isidentifier");
    need_ascii(s, "str.isidentifier");
    if (s.empty() || ascii_digit(s[0])) return Value::boolean(false);
    return Value::boolean(std::all_of(s.begin(), s.end(), [](char32_t x) { return ascii_alpha(x) || ascii_digit(x) || x == '_'; }));
}
inline Value s_islower(Call& c) {
    const U32& s = self_str(c, "islower");
    need_ascii(s, "str.islower");
    bool cased = false;
    for (const char32_t x : s) {
        if (ascii_upper(x)) return Value::boolean(false);
        cased = cased || ascii_lower(x);
    }
    return Value::boolean(cased);
}
inline Value s_isupper(Call& c) {
    const U32& s = self_str(c, "isupper");
    need_ascii(s, "str.isupper");
    bool cased = false;
    for (const char32_t x : s) {
        if (ascii_lower(x)) return Value::boolean(false);
        cased = cased || ascii_upper(x);
    }
    return Value::boolean(cased);
}
inline Value s_istitle(Call& c) {
    const U32& s = self_str(c, "istitle");
    need_ascii(s, "str.istitle");
    bool cased = false, previous = false;
    for (const char32_t x : s) {
        if (ascii_upper(x)) {
            if (previous) return Value::boolean(false);
            previous = cased = true;
        } else if (ascii_lower(x)) {
            if (!previous) return Value::boolean(false);
        } else {
            previous = false;
        }
    }
    return Value::boolean(cased);
}
inline Value strip_impl(Call& c, const char* name, bool left, bool right) {
    const U32& s = self_str(c, name);
    std::size_t a = 0, b = s.size();
    if (c.n > 1 && !c.a[1].is_none()) {
        const U32& chars = str_arg(c.a[1], "strip arg");
        auto in = [&](char32_t x) { return chars.find(x) != U32::npos; };
        rt->tick(s.size() >> 4);
        if (left) while (a < b && in(s[a])) ++a;
        if (right) while (b > a && in(s[b - 1])) --b;
    } else {
        if (left) while (a < b && is_space(s[a])) ++a;
        if (right) while (b > a && is_space(s[b - 1])) --b;
    }
    return slice_text(s, a, b);
}
inline Value s_strip(Call& c) { return strip_impl(c, "strip", true, true); }
inline Value s_lstrip(Call& c) { return strip_impl(c, "lstrip", true, false); }
inline Value s_rstrip(Call& c) { return strip_impl(c, "rstrip", false, true); }
inline char32_t fill_arg(const Call& c, int at) {
    if (c.n <= at) return U' ';
    if (c.a[at].tag != Tag::Str || str_of(c.a[at]).size() != 1) raise(T_TypeError, "The fill character must be exactly one character long");
    return str_of(c.a[at])[0];
}
inline Value s_ljust(Call& c) {
    const U32& s = self_str(c, "ljust");
    return text(pad_to(U32(s), int_arg(c.a[1]), '<', fill_arg(c, 2)));
}
inline Value s_rjust(Call& c) {
    const U32& s = self_str(c, "rjust");
    return text(pad_to(U32(s), int_arg(c.a[1]), '>', fill_arg(c, 2)));
}
inline Value s_center(Call& c) {
    const U32& s = self_str(c, "center");
    const int64_t width = int_arg(c.a[1]);
    const char32_t fill = fill_arg(c, 2);
    const int64_t pad = width - (int64_t) s.size();
    if (pad <= 0) return c.a[0];
    const int64_t left = pad / 2 + (pad & width & 1);
    rt->tick((uint64_t) width >> 6);
    U32 out;
    out.reserve((std::size_t) width);
    out.append((std::size_t) left, fill);
    out += s;
    out.append((std::size_t) (pad - left), fill);
    return text(std::move(out));
}
inline Value s_zfill(Call& c) {
    const U32& s = self_str(c, "zfill");
    const int64_t width = int_arg(c.a[1]);
    const int64_t pad = width - (int64_t) s.size();
    if (pad <= 0) return c.a[0];
    rt->tick((uint64_t) width >> 6);
    const bool sign = !s.empty() && (s[0] == '+' || s[0] == '-');
    U32 out;
    out.reserve((std::size_t) width);
    if (sign) out.push_back(s[0]);
    out.append((std::size_t) pad, U'0');
    out.append(s, sign ? 1 : 0, std::string::npos);
    return text(std::move(out));
}
inline Value s_expandtabs(Call& c) {
    const U32& s = self_str(c, "expandtabs");
    only_keywords(c, {"tabsize"}, "expandtabs");
    Value* kw = c.keyword("tabsize");
    const int64_t size = kw ? int_arg(*kw) : c.n > 1 ? int_arg(c.a[1]) : 8;
    U32 out;
    int64_t column = 0;
    for (const char32_t x : s) {
        if (x == '\t') {
            if (size > 0) {
                const int64_t n = size - column % size;
                rt->tick((uint64_t) n >> 6);
                out.append((std::size_t) n, U' ');
                column += n;
            }
        } else {
            out.push_back(x);
            column = x == '\n' || x == '\r' ? 0 : column + 1;
        }
    }
    return text(std::move(out));
}
inline Value s_partition(Call& c) {
    const U32& s = self_str(c, "partition");
    const U32& d = str_arg(c.a[1], "");
    if (d.empty()) raise(T_ValueError, "empty separator");
    const int64_t hit = find_in(s, d, 0, (int64_t) s.size());
    Vals out;
    if (hit < 0) {
        out.push_back(c.a[0]);
        out.push_back(text(U32{}));
        out.push_back(text(U32{}));
    } else {
        out.push_back(slice_text(s, 0, (std::size_t) hit));
        out.push_back(c.a[1]);
        out.push_back(slice_text(s, (std::size_t) hit + d.size(), s.size()));
    }
    return make_tuple(std::move(out));
}
inline Value s_rpartition(Call& c) {
    const U32& s = self_str(c, "rpartition");
    const U32& d = str_arg(c.a[1], "");
    if (d.empty()) raise(T_ValueError, "empty separator");
    const int64_t hit = rfind_in(s, d, 0, (int64_t) s.size());
    Vals out;
    if (hit < 0) {
        out.push_back(text(U32{}));
        out.push_back(text(U32{}));
        out.push_back(c.a[0]);
    } else {
        out.push_back(slice_text(s, 0, (std::size_t) hit));
        out.push_back(c.a[1]);
        out.push_back(slice_text(s, (std::size_t) hit + d.size(), s.size()));
    }
    return make_tuple(std::move(out));
}
inline Value s_removeprefix(Call& c) {
    const U32& s = self_str(c, "removeprefix");
    const U32& p = str_arg(c.a[1], "removeprefix() argument");
    if (s.size() >= p.size() && s.compare(0, p.size(), p) == 0) return slice_text(s, p.size(), s.size());
    return c.a[0];
}
inline Value s_removesuffix(Call& c) {
    const U32& s = self_str(c, "removesuffix");
    const U32& p = str_arg(c.a[1], "removesuffix() argument");
    if (!p.empty() && s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0) return slice_text(s, 0, s.size() - p.size());
    return c.a[0];
}
inline Value s_maketrans(Call& c) {
    Value out = make_dict();
    Table& t = out.as<DictObj>()->t;
    if (c.n == 1) {
        if (c.a[0].tag != Tag::Dict) raise(T_TypeError, "if you give only one argument to maketrans it must be a dict");
        for (const auto& e : c.a[0].as<DictObj>()->t.entries) {
            if (e.key.empty()) continue;
            if (e.key.tag == Tag::Str) {
                if (str_of(e.key).size() != 1) raise(T_ValueError, "string keys in translate table must be of length 1");
                t.set(Value::integer((int64_t) str_of(e.key)[0]), e.val);
            } else {
                t.set(e.key, e.val);
            }
        }
        return out;
    }
    const U32& x = str_arg(c.a[0], "maketrans() argument 1");
    const U32& y = str_arg(c.a[1], "maketrans() argument 2");
    if (x.size() != y.size()) raise(T_ValueError, "the first two maketrans arguments must have equal length");
    for (std::size_t i = 0; i < x.size(); ++i) t.set(Value::integer((int64_t) x[i]), Value::integer((int64_t) y[i]));
    if (c.n > 2)
        for (const char32_t d : str_arg(c.a[2], "maketrans() argument 3")) t.set(Value::integer((int64_t) d), Value::none());
    return out;
}
inline Value s_translate(Call& c) {
    const U32& s = self_str(c, "translate");
    const Value& table = c.a[1];
    U32 out;
    for (const char32_t x : s) {
        rt->tick();
        Value mapped;
        const Value key = Value::integer((int64_t) x);
        if (table.tag == Tag::Dict) {
            Value* hit = table.as<DictObj>()->t.get(key);
            if (!hit) {
                out.push_back(x);
                continue;
            }
            mapped = *hit;
        } else {
            refuse("str.translate with a table that is not a dict");
        }
        if (mapped.is_none()) continue;
        if (mapped.tag == Tag::Str) out += str_of(mapped);
        else if (is_int(mapped)) out.push_back((char32_t) int_arg(mapped));
        else raise(T_TypeError, "character mapping must return integer, None or str");
    }
    return text(std::move(out));
}
inline Value format_value(const Value& v, const U32& spec) {
    if (spec.empty()) return str_value(v);
    if (v.tag == Tag::Str) {
        Spec sp = parse_spec(spec, "str", true);
        if (sp.type && sp.type != 's') raise(T_ValueError, unknown_code(sp.type, "str"));
        if (sp.sign) raise(T_ValueError, "Sign not allowed in string format specifier");
        if (sp.alt) raise(T_ValueError, "Alternate form (#) not allowed in string format specifier");
        if (sp.group) raise(T_ValueError, std::string("Cannot specify '") + sp.group + "' with 's'.");
        if (sp.align == '=') raise(T_ValueError, "'=' alignment not allowed in string format specifier");
        U32 body = str_of(v);
        if (sp.precision >= 0 && (int64_t) body.size() > sp.precision) body.resize((std::size_t) sp.precision);
        return text(pad_to(std::move(body), sp.width, sp.align && sp.align != '=' ? sp.align : '<', sp.fill));
    }
    if (is_int(v)) return text(format_int_spec(v, parse_spec(spec, "int")));
    if (v.tag == Tag::Float) return text(format_float_spec(v.d, parse_spec(spec, "float")));
    raise(T_TypeError, std::string("unsupported format string passed to ") + type_name(v) + ".__format__");
}
inline Value get_attr(const Value& obj, std::string_view name);
inline Value render_field(const U32& format, std::size_t& i, const Call& c, int64_t& auto_index, int depth);
// The body of str.format: `{field!conv:spec}`, `{{`, `}}`.
inline U32 render_format(const U32& format, const Call& c, int64_t& auto_index, int depth) {
    U32 out;
    std::size_t i = 0;
    while (i < format.size()) {
        const char32_t x = format[i];
        if (x == '{') {
            if (i + 1 < format.size() && format[i + 1] == '{') {
                out.push_back(U'{');
                i += 2;
                continue;
            }
            if (++i >= format.size()) raise(T_ValueError, "Single '{' encountered in format string");
            out += str_of(render_field(format, i, c, auto_index, depth));
        } else if (x == '}') {
            if (i + 1 < format.size() && format[i + 1] == '}') {
                out.push_back(U'}');
                i += 2;
                continue;
            }
            raise(T_ValueError, "Single '}' encountered in format string");
        } else {
            out.push_back(x);
            ++i;
        }
    }
    return out;
}
inline Value render_field(const U32& format, std::size_t& i, const Call& c, int64_t& auto_index, int depth) {
    if (depth > 2) raise(T_ValueError, "Max string recursion exceeded");
    // field_name ends at `!`, `:` or the closing brace, where a `[...]` may hold any of them
    const std::size_t begin = i;
    int nested = 1;
    std::size_t field_end = std::string::npos;
    bool in_index = false;
    for (; i < format.size(); ++i) {
        const char32_t x = format[i];
        if (field_end == std::string::npos) {
            if (x == '[') in_index = true;
            else if (x == ']') in_index = false;
            else if (!in_index && (x == '!' || x == ':')) field_end = i;
        }
        if (x == '{') ++nested;
        else if (x == '}' && --nested == 0) break;
    }
    if (i >= format.size()) raise(T_ValueError, "expected '}' before end of string");
    const std::size_t close = i;
    if (field_end == std::string::npos) field_end = close;
    ++i;
    std::size_t at = begin;
    while (at < field_end && format[at] != '.' && format[at] != '[') ++at;
    const U32 head(format, begin, at - begin);
    Value value;
    if (head.empty()) {
        if (auto_index < 0) raise(T_ValueError, "cannot switch from manual field specification to automatic field numbering");
        const int64_t n = auto_index++;
        if (1 + n >= c.n) raise(T_IndexError, "Replacement index " + std::to_string(n) + " out of range for positional args tuple");
        value = c.a[1 + n];
    } else if (std::all_of(head.begin(), head.end(), ascii_digit)) {
        if (auto_index > 0) raise(T_ValueError, "cannot switch from automatic field numbering to manual field specification");
        auto_index = -1;
        int64_t n = 0;
        for (const char32_t d : head) n = std::min<int64_t>(n * 10 + (d - '0'), INT32_MAX);
        if (1 + n >= c.n) raise(T_IndexError, "Replacement index " + std::to_string(n) + " out of range for positional args tuple");
        value = c.a[1 + n];
    } else {
        const std::string name = utf8(head);
        Value* found = c.keyword(name.c_str());
        if (!found) raise_value(T_KeyError, text(name));
        value = *found;
    }
    while (at < field_end) {
        if (format[at] == '.') {
            std::size_t end = at + 1;
            while (end < field_end && format[end] != '.' && format[end] != '[') ++end;
            const std::string attr = utf8(U32(format, at + 1, end - at - 1));
            if (attr.rfind("__", 0) == 0) refuse("dunder attribute access");
            value = get_attr(value, attr);
            at = end;
        } else {
            const std::size_t end = format.find(U']', at);
            if (end == U32::npos || end > field_end) raise(T_ValueError, "Missing ']' in format string");
            const U32 key(format, at + 1, end - at - 1);
            if (key.empty()) raise(T_ValueError, "Empty attribute in format string");
            if (std::all_of(key.begin(), key.end(), ascii_digit)) {
                int64_t n = 0;
                for (const char32_t d : key) n = std::min<int64_t>(n * 10 + (d - '0'), INT32_MAX);
                value = subscript(value, Value::integer(n));
            } else {
                value = subscript(value, text(U32(key)));
            }
            at = end + 1;
        }
    }
    std::size_t spec_begin = field_end;
    if (spec_begin < close && format[spec_begin] == '!') {
        if (spec_begin + 1 >= close) raise(T_ValueError, "end of string while looking for conversion specifier");
        const char32_t conv = format[spec_begin + 1];
        spec_begin += 2;
        if (conv == 'r') value = text(repr_utf8(value));
        else if (conv == 's') value = str_value(value);
        else if (conv == 'a') {
            const std::string r = repr_utf8(value);
            for (const char ch : r)
                if ((unsigned char) ch >= 128) refuse("ascii() of non-ASCII text");
            value = text(r);
        } else raise(T_ValueError, std::string("Unknown conversion specifier ") + (char) conv);
        if (spec_begin < close && format[spec_begin] != ':') raise(T_ValueError, "expected ':' after conversion specifier");
    }
    U32 spec;
    if (spec_begin < close) {
        const U32 raw(format, spec_begin + 1, close - spec_begin - 1);
        spec = render_format(raw, c, auto_index, depth + 1);
    }
    Value rendered = format_value(value, spec);
    return rendered.tag == Tag::Str ? rendered : str_value(rendered);
}
inline Value s_format(Call& c) {
    const U32& format = self_str(c, "format");
    int64_t auto_index = 0;
    return text(render_format(format, c, auto_index, 0));
}
inline Value format_percent(const U32& f, const Value& args) {
    U32 out;
    const bool is_tuple = args.tag == Tag::Tuple;
    const std::size_t count = is_tuple ? seq_of(args).size() : 1;
    std::size_t next = 0;
    auto take = [&]() -> Value {
        if (next >= count) raise(T_TypeError, "not enough arguments for format string");
        ++next;
        return is_tuple ? seq_of(args)[next - 1] : args;
    };
    const bool mapping = !is_tuple && args.tag != Tag::Str && (args.tag == Tag::Dict || args.tag == Tag::List || args.tag == Tag::Range || args.tag == Tag::Deque);
    bool used_key = false;
    std::size_t i = 0;
    while (i < f.size()) {
        if (f[i] != '%') {
            out.push_back(f[i++]);
            continue;
        }
        if (++i >= f.size()) raise(T_ValueError, "incomplete format");
        Value key_arg;
        if (f[i] == '(') {
            if (args.tag != Tag::Dict) raise(T_TypeError, "format requires a mapping");
            std::size_t close = i + 1;
            int level = 1;
            for (; close < f.size(); ++close) {
                if (f[close] == '(') ++level;
                else if (f[close] == ')' && --level == 0) break;
            }
            if (close >= f.size()) raise(T_ValueError, "incomplete format key");
            key_arg = subscript(args, text(U32(f, i + 1, close - i - 1)));
            used_key = true;
            i = close + 1;
        }
        Spec sp;
        bool left = false, plus = false, space = false, alt = false, zero = false;
        for (; i < f.size(); ++i) {
            if (f[i] == '-') left = true;
            else if (f[i] == '+') plus = true;
            else if (f[i] == ' ') space = true;
            else if (f[i] == '#') alt = true;
            else if (f[i] == '0') zero = true;
            else break;
        }
        auto number = [&](int64_t& n) {
            if (i < f.size() && f[i] == '*') {
                ++i;
                const Value v = take();
                if (!is_int(v)) raise(T_TypeError, "* wants int");
                n = int_arg(v);
                if (n < 0) {
                    left = true;
                    n = -n;
                }
                return;
            }
            bool any = false;
            int64_t v = 0;
            while (i < f.size() && ascii_digit(f[i])) {
                v = std::min<int64_t>(v * 10 + (f[i++] - '0'), INT32_MAX);
                any = true;
            }
            if (any) n = v;
        };
        number(sp.width);
        int64_t precision = -1;
        if (i < f.size() && f[i] == '.') {
            ++i;
            precision = 0;
            number(precision);
        }
        while (i < f.size() && (f[i] == 'h' || f[i] == 'l' || f[i] == 'L')) ++i;
        if (i >= f.size()) raise(T_ValueError, "incomplete format");
        const char32_t conv = f[i++];
        if (conv == '%') {
            if (left || plus || space || alt || zero || sp.width || precision >= 0) refuse("%% with flags or a width");
            out.push_back(U'%');
            continue;
        }
        const Value v = key_arg.empty() ? take() : key_arg;
        sp.alt = alt;
        sp.sign = plus ? '+' : space ? ' ' : 0;
        sp.align = left ? '<' : '>';
        if (zero && !left) {
            sp.fill = '0';
            sp.align = '=';
        }
        switch (conv) {
            case 's': case 'r': case 'a': {
                U32 body;
                if (conv == 's') body = str_of(str_value(v));
                else {
                    repr_to(body, v);
                    if (conv == 'a') need_ascii(body, "ascii()");
                }
                if (precision >= 0 && (int64_t) body.size() > precision) body.resize((std::size_t) precision);
                out += pad_to(std::move(body), sp.width, left ? '<' : '>', U' ');
                break;
            }
            case 'c': {
                char32_t ch = 0;
                if (v.tag == Tag::Str) {
                    if (str_of(v).size() != 1) raise(T_TypeError, "%c requires int or char");
                    ch = str_of(v)[0];
                } else {
                    if (!is_int(v)) raise(T_TypeError, "%c requires int or char");
                    const int64_t code = int_arg(v);
                    if (code < 0 || code > 0x10FFFF) raise(T_OverflowError, "%c arg not in range(0x110000)");
                    ch = (char32_t) code;
                }
                out += pad_to(U32(1, ch), sp.width, left ? '<' : '>', U' ');
                break;
            }
            case 'd': case 'i': case 'u': case 'o': case 'x': case 'X': {
                Value number_value = v;
                const bool decimal = conv == 'd' || conv == 'i' || conv == 'u';
                if (v.tag == Tag::Float && decimal) {
                    if (std::isinf(v.d)) raise(T_OverflowError, "cannot convert float infinity to integer");
                    if (std::isnan(v.d)) raise(T_ValueError, "cannot convert float NaN to integer");
                    number_value = int_from_double(std::trunc(v.d));
                } else if (v.tag == Tag::Fraction && decimal) {
                    auto* q = v.as<FractionObj>();
                    number_value = int_is_neg(q->num) ? int_negate(int_floordiv(int_negate(q->num), q->den)) : int_floordiv(q->num, q->den);
                }
                if (!is_int(number_value)) raise(T_TypeError, std::string("%") + (char) conv + " format: " + (decimal ? "a real number" : "an integer") + " is required, not " + type_name(v));
                std::string digits = decimal ? int_decimal(int_abs(number_value)) : int_base(number_value, conv == 'o' ? 8 : 16);
                if (conv == 'X')
                    for (auto& ch : digits) ch = (char) std::toupper((unsigned char) ch);
                if (precision > (int64_t) digits.size()) digits.insert(0, (std::size_t) (precision - (int64_t) digits.size()), '0');
                std::string head = int_is_neg(number_value) ? "-" : sp.sign ? std::string(1, sp.sign) : "";
                if (alt && !decimal) head += std::string("0") + (char) (conv == 'o' ? 'o' : conv);
                sp.alt = false;
                out += lay_out_number(sp, head, digits, "");
                break;
            }
            case 'e': case 'E': case 'f': case 'F': case 'g': case 'G': {
                if (!is_number(v) && v.tag != Tag::Fraction) raise(T_TypeError, std::string("must be real number, not ") + type_name(v));
                sp.type = (char) conv;
                sp.precision = precision < 0 ? 6 : precision;
                out += format_float_spec(to_float(v), sp);
                break;
            }
            default: raise(T_ValueError, std::string("unsupported format character '") + utf8(U32(1, conv)) + "' (0x" + int_base(Value::integer((int64_t) conv), 16) + ") at index " + std::to_string(i - 1));
        }
    }
    if (next < count && !used_key && !mapping) raise(T_TypeError, "not all arguments converted during string formatting");
    return text(std::move(out));
}

inline void list_extend(ListObj* list, const Value& items) {
    if (items.tag == Tag::List || items.tag == Tag::Tuple) {
        const Vals copy = seq_of(items);
        rt->tick(copy.size() >> 2);
        list->v.insert(list->v.end(), copy.begin(), copy.end());
        return;
    }
    each(items, [&](Value& x) { list->v.push_back(std::move(x)); });
}
inline ListObj* self_list(const Call& c, const char* name) {
    if (c.a[0].tag != Tag::List) raise(T_TypeError, std::string("descriptor '") + name + "' for 'list' objects doesn't apply to a " + quoted(c.a[0]) + " object");
    return c.a[0].as<ListObj>();
}
inline Value l_append(Call& c) {
    self_list(c, "append")->v.push_back(c.a[1]);
    return Value::none();
}
inline Value l_extend(Call& c) {
    list_extend(self_list(c, "extend"), c.a[1]);
    return Value::none();
}
inline Value l_clear(Call& c) {
    Vals gone;
    gone.swap(self_list(c, "clear")->v);
    return Value::none();
}
inline Value l_insert(Call& c) {
    Vals& v = self_list(c, "insert")->v;
    int64_t at = int_arg(c.a[1]);
    const int64_t n = (int64_t) v.size();
    if (at < 0) at = std::max<int64_t>(at + n, 0);
    if (at > n) at = n;
    rt->tick((uint64_t) n >> 4);
    v.insert(v.begin() + at, c.a[2]);
    return Value::none();
}
inline Value l_pop(Call& c) {
    Vals& v = self_list(c, "pop")->v;
    if (v.empty()) raise(T_IndexError, "pop from empty list");
    int64_t at = c.n > 1 ? int_arg(c.a[1]) : -1;
    const int64_t n = (int64_t) v.size();
    if (at < 0) at += n;
    if (at < 0 || at >= n) raise(T_IndexError, "pop index out of range");
    Value out = std::move(v[(std::size_t) at]);
    rt->tick((uint64_t) (n - at) >> 4);
    v.erase(v.begin() + at);
    return out;
}
inline Value l_remove(Call& c) {
    Vals& v = self_list(c, "remove")->v;
    for (std::size_t i = 0; i < v.size(); ++i) {
        rt->tick();
        if (same(v[i], c.a[1])) {
            Value keep = std::move(v[i]);
            v.erase(v.begin() + (int64_t) i);
            return Value::none();
        }
    }
    raise(T_ValueError, "list.remove(x): x not in list");
}
inline Value l_reverse(Call& c) {
    Vals& v = self_list(c, "reverse")->v;
    rt->tick(v.size() >> 4);
    std::reverse(v.begin(), v.end());
    return Value::none();
}
inline Value l_copy(Call& c) {
    Vals v = self_list(c, "copy")->v;
    rt->tick(v.size() >> 2);
    return make_list(std::move(v));
}
inline Value seq_count(const Vals& v, const Value& x) {
    int64_t n = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        rt->tick();
        if (same(v[i], x)) ++n;
    }
    return Value::integer(n);
}
inline Value seq_find(const Vals& v, Call& c, const char* missing) {
    const Bounds b = bounds_arg(c, 2, (int64_t) v.size());
    for (int64_t i = b.start; i < b.stop && i < (int64_t) v.size(); ++i) {
        rt->tick();
        if (same(v[(std::size_t) i], c.a[1])) return Value::integer(i);
    }
    raise(T_ValueError, missing);
}
inline Value l_count(Call& c) { return seq_count(self_list(c, "count")->v, c.a[1]); }
inline Value l_index(Call& c) { return seq_find(self_list(c, "index")->v, c, (repr_utf8(c.a[1]) + " is not in list").c_str()); }
inline Value t_count(Call& c) { return seq_count(seq_of(c.a[0]), c.a[1]); }
inline Value t_index(Call& c) { return seq_find(seq_of(c.a[0]), c, "tuple.index(x): x not in tuple"); }
// Comparison sorts agree only on a total order. NaN and sets are not one, and the answer would depend on Timsort's exact steps.
inline void require_total_order(const Value& v) {
    if (v.tag == Tag::Float && std::isnan(v.d)) refuse("sorting with nan");
    if (v.tag == Tag::Set) refuse("sorting sets");
    if (v.tag == Tag::List || v.tag == Tag::Tuple) {
        Nest nest;
        for (const auto& item : seq_of(v)) require_total_order(item);
    }
}
inline void sort_values(Vals& items, const Value& key, bool reverse, bool* comparing = nullptr) {
    Vals keys;
    if (!key.is_none()) {
        keys.reserve(items.size());
        for (auto& item : items) {
            Value arg = item;
            rt->tick();
            keys.push_back(call_value(key, &arg, 1));
        }
    }
    const Vals& by = key.is_none() ? items : keys;
    for (const auto& k : by) require_total_order(k);
    std::vector<uint32_t, Alloc<uint32_t>> order_index(items.size());
    for (std::size_t i = 0; i < order_index.size(); ++i) order_index[i] = (uint32_t) i;
    rt->charge(items.size() * sizeof(uint32_t));   // stable_sort's own buffer
    struct Hold {
        std::size_t n;
        ~Hold() { rt->release(n); }
    } hold{items.size() * sizeof(uint32_t)};
    const Cmp op = reverse ? Cmp::Gt : Cmp::Lt;
    if (comparing) *comparing = true;
    std::stable_sort(order_index.begin(), order_index.end(), [&](uint32_t x, uint32_t y) {
        rt->tick();
        return order(op, by[x], by[y]);
    });
    Vals out;
    out.reserve(items.size());
    for (const uint32_t i : order_index) out.push_back(std::move(items[i]));
    items.swap(out);
}
inline Value l_sort(Call& c) {
    auto* list = self_list(c, "sort");
    only_keywords(c, {"key", "reverse"}, "sort");
    Value* key = c.keyword("key");
    Value* rev = c.keyword("reverse");
    // CPython empties the list while it sorts, and a key function that sees it there sees it empty.
    Vals items;
    items.swap(list->v);
    bool comparing = false;
    Value failed;
    try {
        sort_values(items, key ? *key : Value::none(), rev ? truth(*rev) : false, &comparing);
    } catch (const PyError& e) {
        failed = e.exc;
    }
    if (!failed.empty()) {
        // A failed compare leaves a partly sorted list, in an order only Timsort's steps can give.
        if (comparing) refuse("list.sort that raised partway: " + utf8(str_of(str_value(failed))));
        list->v.swap(items);
        throw PyError{std::move(failed)};
    }
    const bool touched = !list->v.empty();
    list->v.swap(items);
    if (touched) raise(T_ValueError, "list modified during sort");
    return Value::none();
}

inline DictObj* self_dict(const Call& c, const char* name) {
    if (c.a[0].tag != Tag::Dict) raise(T_TypeError, std::string("descriptor '") + name + "' for 'dict' objects doesn't apply to a " + quoted(c.a[0]) + " object");
    return c.a[0].as<DictObj>();
}
inline void dict_update_pairs(DictObj* d, const Value& source) {
    int64_t index = 0;
    each(source, [&](Value& item) {
        Vals parts;
        if (item.tag != Tag::List && item.tag != Tag::Tuple && item.tag != Tag::Str && item.tag != Tag::Iter && item.tag != Tag::Set && item.tag != Tag::Dict && item.tag != Tag::Range)
            raise(T_TypeError, "cannot convert dictionary update sequence element #" + std::to_string(index) + " to a sequence");
        parts = to_vals(item);
        if (parts.size() != 2) raise(T_ValueError, "dictionary update sequence element #" + std::to_string(index) + " has length " + std::to_string(parts.size()) + "; 2 is required");
        d->t.set(parts[0], parts[1]);
        ++index;
    });
}
inline void dict_update_from(DictObj* d, const Value& source) {
    if (source.tag == Tag::Dict) {
        const Table& t = source.as<DictObj>()->t;
        rt->tick(t.used);
        for (std::size_t i = 0; i < t.entries.size(); ++i) {
            if (t.entries[i].key.empty()) continue;
            const Value key = t.entries[i].key, val = t.entries[i].val;
            d->t.set(key, val);
        }
        return;
    }
    dict_update_pairs(d, source);
}
inline Value dict_copy(const Value& d) {
    auto* from = d.as<DictObj>();
    Value out = make_dict(from->kind);
    out.as<DictObj>()->factory = from->factory;
    dict_update_from(out.as<DictObj>(), d);
    return out;
}
inline Value view(const Value& d, int type) { return Value(new ViewObj(d, type)); }
inline Value d_get(Call& c) {
    Value* found = self_dict(c, "get")->t.get(c.a[1]);
    if (found) return *found;
    return c.n > 2 ? c.a[2] : Value::none();
}
inline Value d_keys(Call& c) { self_dict(c, "keys"); return view(c.a[0], T_dict_keys); }
inline Value d_values(Call& c) { self_dict(c, "values"); return view(c.a[0], T_dict_values); }
inline Value d_items(Call& c) { self_dict(c, "items"); return view(c.a[0], T_dict_items); }
inline Value d_pop(Call& c) {
    Value out;
    if (self_dict(c, "pop")->t.erase(c.a[1], &out)) return out;
    if (c.n > 2) return c.a[2];
    raise_value(T_KeyError, c.a[1]);
}
inline Value d_popitem(Call& c) {
    Table& t = self_dict(c, "popitem")->t;
    while (!t.entries.empty() && t.entries.back().key.empty()) t.entries.pop_back();
    if (t.entries.empty()) raise_value(T_KeyError, text("popitem(): dictionary is empty"));
    const Value key = t.entries.back().key;
    Value val;
    t.erase(key, &val);
    t.entries.pop_back();
    return pair(key, std::move(val));
}
inline Value d_setdefault(Call& c) {
    Table& t = self_dict(c, "setdefault")->t;
    if (Value* found = t.get(c.a[1])) return *found;
    const Value fallback = c.n > 2 ? c.a[2] : Value::none();
    t.set(c.a[1], fallback);
    return fallback;
}
inline void counter_add(DictObj* d, const Value& source, int sign);
inline Value d_update(Call& c) {
    auto* d = self_dict(c, "update");
    if (c.n > 1) {
        if (d->kind == DictKind::Counter) counter_add(d, c.a[1], 1);
        else dict_update_from(d, c.a[1]);
    }
    for (int k = 0; k < c.nkw(); ++k) {
        if (d->kind == DictKind::Counter) {
            Value* have = d->t.get(c.kw_names->v[(std::size_t) k]);
            d->t.set(c.kw_names->v[(std::size_t) k], binary(Bin::Add, have ? *have : Value::integer(0), c.a[c.n + k], false));
        } else {
            d->t.set(c.kw_names->v[(std::size_t) k], c.a[c.n + k]);
        }
    }
    return Value::none();
}
inline Value d_clear(Call& c) {
    Table gone;
    std::swap(gone.entries, self_dict(c, "clear")->t.entries);
    std::swap(gone.index, c.a[0].as<DictObj>()->t.index);
    c.a[0].as<DictObj>()->t.used = 0;
    return Value::none();
}
inline Value d_copy(Call& c) { self_dict(c, "copy"); return dict_copy(c.a[0]); }
inline Value d_fromkeys(Call& c) {
    Value out = make_dict();
    const Value fill = c.n > 1 ? c.a[1] : Value::none();
    each(c.a[0], [&](Value& k) { out.as<DictObj>()->t.set(k, fill); });
    return out;
}
inline void counter_add(DictObj* d, const Value& source, int sign) {
    if (source.tag == Tag::Dict) {
        const Table& t = source.as<DictObj>()->t;
        for (std::size_t i = 0; i < t.entries.size(); ++i) {
            if (t.entries[i].key.empty()) continue;
            const Value key = t.entries[i].key, val = t.entries[i].val;
            Value* have = d->t.get(key);
            const Value base = have ? *have : Value::integer(0);
            d->t.set(key, binary(sign > 0 ? Bin::Add : Bin::Sub, base, val, false));
        }
        return;
    }
    each(source, [&](Value& k) {
        Value* have = d->t.get(k);
        const Value base = have ? *have : Value::integer(0);
        d->t.set(k, binary(sign > 0 ? Bin::Add : Bin::Sub, base, Value::integer(1), false));
    });
}
inline Vals counter_sorted(DictObj* d) {
    Vals items, keys;
    for (std::size_t i = 0; i < d->t.entries.size(); ++i) {
        if (d->t.entries[i].key.empty()) continue;
        items.push_back(pair(d->t.entries[i].key, d->t.entries[i].val));
    }
    std::vector<uint32_t, Alloc<uint32_t>> at(items.size());
    for (std::size_t i = 0; i < at.size(); ++i) at[i] = (uint32_t) i;
    std::stable_sort(at.begin(), at.end(), [&](uint32_t x, uint32_t y) {
        rt->tick();
        return order(Cmp::Gt, seq_of(items[x])[1], seq_of(items[y])[1]);
    });
    Vals out;
    out.reserve(items.size());
    for (const uint32_t i : at) out.push_back(items[i]);
    return out;
}
inline Value c_most_common(Call& c) {
    Vals items = counter_sorted(self_dict(c, "most_common"));
    if (c.n > 1 && !c.a[1].is_none()) {
        const int64_t n = int_arg(c.a[1]);
        if (n < (int64_t) items.size()) items.resize((std::size_t) std::max<int64_t>(n, 0));
    }
    return make_list(std::move(items));
}
inline Value c_elements(Call& c) {
    auto* d = self_dict(c, "elements");
    Vals out;
    for (const auto& e : d->t.entries) {
        if (e.key.empty()) continue;
        if (!is_int(e.val)) raise(T_TypeError, "elements needs integer counts");
        const int64_t n = clamp_index(e.val);
        for (int64_t i = 0; i < n; ++i) {
            rt->tick();
            out.push_back(e.key);
        }
    }
    return get_iter(make_list(std::move(out)));
}
inline Value c_subtract(Call& c) {
    auto* d = self_dict(c, "subtract");
    if (c.n > 1) counter_add(d, c.a[1], -1);
    return Value::none();
}
[[noreturn]] inline Value c_fromkeys(Call&) { raise(T_NotImplementedError, "Counter.fromkeys() is undefined.  Use Counter(iterable) instead."); }
inline Value c_total(Call& c) {
    Value sum = Value::integer(0);
    for (const auto& e : self_dict(c, "total")->t.entries)
        if (!e.key.empty()) sum = binary(Bin::Add, sum, e.val, false);
    return sum;
}
inline Value counter_binary(Bin op, const Value& a, const Value& b) {
    Value out = make_dict(DictKind::Counter);
    Table& result = out.as<DictObj>()->t;
    Table& x = a.as<DictObj>()->t;
    Table& y = b.as<DictObj>()->t;
    const Value zero = Value::integer(0);
    for (const auto& e : x.entries) {
        if (e.key.empty()) continue;
        Value* other_found = y.get(e.key);
        const Value other = other_found ? *other_found : zero;
        Value next;
        if (op == Bin::Add) next = binary(Bin::Add, e.val, other, false);
        else if (op == Bin::Sub) next = binary(Bin::Sub, e.val, other, false);
        else if (op == Bin::Or) next = order(Cmp::Lt, e.val, other) ? other : e.val;
        else if (op == Bin::And) next = order(Cmp::Lt, e.val, other) ? e.val : other;
        else bad_operands(op, a, b, false);
        if (order(Cmp::Gt, next, zero)) result.set(e.key, next);
    }
    if (op == Bin::And) return out;
    for (const auto& e : y.entries) {
        if (e.key.empty() || x.get(e.key)) continue;
        if (op == Bin::Sub) {
            if (order(Cmp::Lt, e.val, zero)) result.set(e.key, binary(Bin::Sub, zero, e.val, false));
        } else if (order(Cmp::Gt, e.val, zero)) {
            result.set(e.key, e.val);
        }
    }
    return out;
}
inline Value counter_result(Bin op, const Value& a, const Value& b, bool inplace) {
    Value out = counter_binary(op, a, b);
    if (!inplace) return out;
    std::swap(a.as<DictObj>()->t.entries, out.as<DictObj>()->t.entries);
    std::swap(a.as<DictObj>()->t.index, out.as<DictObj>()->t.index);
    std::swap(a.as<DictObj>()->t.used, out.as<DictObj>()->t.used);
    ++a.as<DictObj>()->t.rebuilds;
    return a;
}

inline Value items_to_set(const Value& v) {
    Value out = make_set();
    each(v, [&](Value& x) { out.as<SetObj>()->t.set(x, Value::none()); });
    return out;
}
inline Table& set_like(const Value& v, Value& holder) {
    if (v.tag == Tag::Set) return v.as<SetObj>()->t;
    if (v.tag == Tag::Dict) return v.as<DictObj>()->t;
    holder = items_to_set(v);
    return holder.as<SetObj>()->t;
}
inline Value set_copy(const Table& t, bool frozen) {
    Value out = make_set(frozen);
    Table& into = out.as<SetObj>()->t;
    rt->tick(t.used);
    for (const auto& e : t.entries)
        if (!e.key.empty()) into.set(e.key, Value::none());
    return out;
}
inline Value set_binary(Bin op, const Value& a, const Value& b, bool inplace) {
    Value hold_a, hold_b;
    Table& x = set_like(a, hold_a);
    Table& y = set_like(b, hold_b);
    const bool frozen = a.tag == Tag::Set && a.as<SetObj>()->frozen;
    Value out = make_set(frozen);
    Table& r = out.as<SetObj>()->t;
    rt->tick(x.used + y.used);
    for (const auto& e : x.entries) {
        if (e.key.empty()) continue;
        const bool in_y = y.get(e.key) != nullptr;
        if (op == Bin::Or || (op == Bin::And && in_y) || ((op == Bin::Sub || op == Bin::Xor) && !in_y)) r.set(e.key, Value::none());
    }
    if (op == Bin::Or || op == Bin::Xor)
        for (const auto& e : y.entries)
            if (!e.key.empty() && (op == Bin::Or || !x.get(e.key))) r.set(e.key, Value::none());
    if (inplace && a.tag == Tag::Set && !frozen) {
        std::swap(x.entries, r.entries);
        std::swap(x.index, r.index);
        std::swap(x.used, r.used);
        ++x.rebuilds;
        return a;
    }
    return out;
}
inline SetObj* self_set(const Call& c, const char* name) {
    if (c.a[0].tag != Tag::Set) raise(T_TypeError, std::string("descriptor '") + name + "' for 'set' objects doesn't apply to a " + quoted(c.a[0]) + " object");
    return c.a[0].as<SetObj>();
}
inline Value as_set(const Value& v) {
    if (v.tag == Tag::Set) return v;
    return items_to_set(v);
}
inline Value st_add(Call& c) {
    self_set(c, "add")->t.set(c.a[1], Value::none());
    return Value::none();
}
inline Value st_discard(Call& c) {
    self_set(c, "discard")->t.erase(c.a[1]);
    return Value::none();
}
inline Value st_remove(Call& c) {
    if (!self_set(c, "remove")->t.erase(c.a[1])) raise_value(T_KeyError, c.a[1]);
    return Value::none();
}
inline Value st_pop(Call& c) {
    Table& t = self_set(c, "pop")->t;
    for (const auto& e : t.entries) {
        if (e.key.empty()) continue;
        const Value key = e.key;
        t.erase(key);
        return key;
    }
    raise_value(T_KeyError, text("pop from an empty set"));
}
inline Value st_clear(Call& c) {
    Table gone;
    std::swap(gone.entries, self_set(c, "clear")->t.entries);
    std::swap(gone.index, c.a[0].as<SetObj>()->t.index);
    c.a[0].as<SetObj>()->t.used = 0;
    return Value::none();
}
inline Value st_copy(Call& c) {
    auto* s = self_set(c, "copy");
    return set_copy(s->t, s->frozen);
}
inline Value set_fold(Call& c, const char* name, Bin op) {
    auto* s = self_set(c, name);
    Value out = set_copy(s->t, s->frozen);
    for (int i = 1; i < c.n; ++i) out = set_binary(op, out, as_set(c.a[i]), false);
    return out;
}
inline Value st_union(Call& c) { return set_fold(c, "union", Bin::Or); }
inline Value st_intersection(Call& c) { return set_fold(c, "intersection", Bin::And); }
inline Value st_difference(Call& c) { return set_fold(c, "difference", Bin::Sub); }
inline Value st_symmetric_difference(Call& c) { return set_fold(c, "symmetric_difference", Bin::Xor); }
inline Value set_fold_in_place(Call& c, const char* name, Bin op) {
    self_set(c, name);
    for (int i = 1; i < c.n; ++i) set_binary(op, c.a[0], as_set(c.a[i]), true);
    return Value::none();
}
inline Value st_update(Call& c) { return set_fold_in_place(c, "update", Bin::Or); }
inline Value st_intersection_update(Call& c) { return set_fold_in_place(c, "intersection_update", Bin::And); }
inline Value st_difference_update(Call& c) { return set_fold_in_place(c, "difference_update", Bin::Sub); }
inline Value st_symmetric_difference_update(Call& c) { return set_fold_in_place(c, "symmetric_difference_update", Bin::Xor); }
inline Value st_isdisjoint(Call& c) {
    auto* s = self_set(c, "isdisjoint");
    Value other = as_set(c.a[1]);
    for (const auto& e : other.as<SetObj>()->t.entries)
        if (!e.key.empty() && s->t.get(e.key)) return Value::boolean(false);
    return Value::boolean(true);
}
inline Value st_issubset(Call& c) {
    auto* s = self_set(c, "issubset");
    Value other = as_set(c.a[1]);
    return Value::boolean(table_subset(s->t, other.as<SetObj>()->t));
}
inline Value st_issuperset(Call& c) {
    auto* s = self_set(c, "issuperset");
    Value other = as_set(c.a[1]);
    return Value::boolean(table_subset(other.as<SetObj>()->t, s->t));
}

inline void deque_append(DequeObj* q, Value v) {
    ++q->version;
    if (q->maxlen == 0) return;
    if (q->maxlen > 0 && (int64_t) q->count == q->maxlen) {
        q->at(0).reset();
        q->head = (q->head + 1) % q->ring.size();
        --q->count;
    }
    q->grow();
    q->ring[(q->head + q->count) % q->ring.size()] = std::move(v);
    ++q->count;
}
inline void deque_appendleft(DequeObj* q, Value v) {
    ++q->version;
    if (q->maxlen == 0) return;
    if (q->maxlen > 0 && (int64_t) q->count == q->maxlen) {
        q->at(q->count - 1).reset();
        --q->count;
    }
    q->grow();
    q->head = (q->head + q->ring.size() - 1) % q->ring.size();
    q->ring[q->head] = std::move(v);
    ++q->count;
}
inline Value deque_pop(DequeObj* q) {
    if (!q->count) raise(T_IndexError, "pop from an empty deque");
    ++q->version;
    Value out = std::move(q->at(q->count - 1));
    --q->count;
    return out;
}
inline Value deque_popleft(DequeObj* q) {
    if (!q->count) raise(T_IndexError, "pop from an empty deque");
    ++q->version;
    Value out = std::move(q->at(0));
    q->head = (q->head + 1) % q->ring.size();
    --q->count;
    return out;
}
inline Value deque_copy(const Value& d) {
    auto* from = d.as<DequeObj>();
    auto* to = new DequeObj;
    Value out(to);
    to->maxlen = from->maxlen;
    for (std::size_t i = 0; i < from->count; ++i) deque_append(to, Value(from->at(i)));
    return out;
}
inline DequeObj* self_deque(const Call& c) {
    if (c.a[0].tag != Tag::Deque) raise(T_TypeError, "requires a 'collections.deque' object");
    return c.a[0].as<DequeObj>();
}
inline Value q_append(Call& c) { deque_append(self_deque(c), c.a[1]); return Value::none(); }
inline Value q_appendleft(Call& c) { deque_appendleft(self_deque(c), c.a[1]); return Value::none(); }
inline Value q_pop(Call& c) { return deque_pop(self_deque(c)); }
inline Value q_popleft(Call& c) { return deque_popleft(self_deque(c)); }
inline Value q_extend(Call& c) {
    auto* q = self_deque(c);
    const Vals items = to_vals(c.a[1]);
    for (const auto& x : items) deque_append(q, x);
    return Value::none();
}
inline Value q_extendleft(Call& c) {
    auto* q = self_deque(c);
    const Vals items = to_vals(c.a[1]);
    for (const auto& x : items) deque_appendleft(q, x);
    return Value::none();
}
inline Value q_clear(Call& c) {
    auto* q = self_deque(c);
    ++q->version;
    Vals gone;
    gone.swap(q->ring);
    q->head = q->count = 0;
    return Value::none();
}
inline Value q_copy(Call& c) { self_deque(c); return deque_copy(c.a[0]); }
inline Value q_count(Call& c) {
    auto* q = self_deque(c);
    int64_t n = 0;
    for (std::size_t i = 0; i < q->count; ++i) {
        rt->tick();
        if (same(q->at(i), c.a[1])) ++n;
    }
    return Value::integer(n);
}
inline Value q_index(Call& c) {
    auto* q = self_deque(c);
    const Bounds b = bounds_arg(c, 2, (int64_t) q->count);
    for (int64_t i = b.start; i < b.stop; ++i) {
        rt->tick();
        if (same(q->at((std::size_t) i), c.a[1])) return Value::integer(i);
    }
    raise(T_ValueError, repr_utf8(c.a[1]) + " is not in deque");
}
inline Value q_remove(Call& c) {
    auto* q = self_deque(c);
    for (std::size_t i = 0; i < q->count; ++i) {
        rt->tick();
        if (same(q->at(i), c.a[1])) {
            delete_subscript(c.a[0], Value::integer((int64_t) i));
            return Value::none();
        }
    }
    raise(T_ValueError, repr_utf8(c.a[1]) + " is not in deque");
}
inline Value q_insert(Call& c) {
    auto* q = self_deque(c);
    if (q->maxlen >= 0 && (int64_t) q->count == q->maxlen) raise(T_IndexError, "deque already at its maximum size");
    int64_t at = int_arg(c.a[1]);
    const int64_t n = (int64_t) q->count;
    if (at < 0) at = std::max<int64_t>(at + n, 0);
    if (at > n) at = n;
    deque_append(q, Value::none());
    rt->tick((uint64_t) n >> 4);
    for (int64_t i = n; i > at; --i) q->at((std::size_t) i) = std::move(q->at((std::size_t) i - 1));
    q->at((std::size_t) at) = c.a[2];
    return Value::none();
}
inline Value q_reverse(Call& c) {
    auto* q = self_deque(c);
    ++q->version;
    for (std::size_t i = 0, j = q->count; i + 1 < j; ++i, --j) std::swap(q->at(i), q->at(j - 1));
    return Value::none();
}
inline Value q_rotate(Call& c) {
    auto* q = self_deque(c);
    int64_t n = c.n > 1 ? int_arg(c.a[1]) : 1;
    if (!q->count) return Value::none();
    const int64_t size = (int64_t) q->count;
    n = ((n % size) + size) % size;
    ++q->version;
    rt->tick((uint64_t) std::min(n, size - n));
    for (int64_t i = 0; i < n; ++i) {
        Value last = std::move(q->at(q->count - 1));
        --q->count;
        q->head = (q->head + q->ring.size() - 1) % q->ring.size();
        q->ring[q->head] = std::move(last);
        ++q->count;
    }
    return Value::none();
}

inline Value frac_make(Value n, Value d) {
    if (int_is_zero(d)) raise(T_ZeroDivisionError, "Fraction(" + int_decimal(n) + ", 0)");
    const Value g = int_gcd(n, d);
    n = int_floordiv(n, g);
    d = int_floordiv(d, g);
    if (int_is_neg(d)) {
        n = int_negate(n);
        d = int_negate(d);
    }
    return Value(new FractionObj(std::move(n), std::move(d)));
}
inline Value frac_from_double(double x) {
    if (std::isnan(x)) raise(T_ValueError, "cannot convert NaN to integer ratio");
    if (std::isinf(x)) raise(T_OverflowError, "cannot convert Infinity to integer ratio");
    int e = 0;
    double m = std::frexp(x, &e);
    for (int i = 0; i < 300 && m != std::floor(m); ++i) {
        m *= 2;
        --e;
    }
    Value n = int_from_double(m);
    if (e > 0) return frac_make(int_shl(n, (uint64_t) e), Value::integer(1));
    return frac_make(n, int_shl(Value::integer(1), (uint64_t) -e));
}
inline const Value& frac_num(const Value& v) { return v.tag == Tag::Fraction ? v.as<FractionObj>()->num : v; }
inline Value frac_den(const Value& v) { return v.tag == Tag::Fraction ? v.as<FractionObj>()->den : Value::integer(1); }
inline int frac_cmp(const Value& a, const Value& b) { return int_cmp(int_mul(frac_num(a), frac_den(b)), int_mul(frac_num(b), frac_den(a))); }
inline Value frac_binary(Bin op, const Value& a, const Value& b) {
    if (a.tag == Tag::Float || b.tag == Tag::Float) return number_binary(op, Value::real(to_float(a)), Value::real(to_float(b)), false);
    const Value& n1 = frac_num(a);
    const Value& n2 = frac_num(b);
    const Value d1 = frac_den(a), d2 = frac_den(b);
    switch (op) {
        case Bin::Add: return frac_make(int_add(int_mul(n1, d2), int_mul(n2, d1)), int_mul(d1, d2));
        case Bin::Sub: return frac_make(int_sub(int_mul(n1, d2), int_mul(n2, d1)), int_mul(d1, d2));
        case Bin::Mul: return frac_make(int_mul(n1, n2), int_mul(d1, d2));
        case Bin::Div:
            if (int_is_zero(n2)) {
                // The numerator Fraction._div would have reached, for the message.
                const Value g1 = int_gcd(n1, n2), g2 = int_gcd(d2, d1);
                const Value na = int_cmp(g1, Value::integer(1)) > 0 ? int_floordiv(n1, g1) : n1;
                const Value db = int_cmp(g2, Value::integer(1)) > 0 ? int_floordiv(d2, g2) : d2;
                raise(T_ZeroDivisionError, "Fraction(" + int_decimal(int_mul(na, db)) + ", 0)");
            }
            return frac_make(int_mul(n1, d2), int_mul(d1, n2));
        case Bin::FloorDiv:
            if (int_is_zero(n2)) raise(T_ZeroDivisionError, "integer division or modulo by zero");
            return int_floordiv(int_mul(n1, d2), int_mul(d1, n2));
        case Bin::Mod:
            if (int_is_zero(n2)) raise(T_ZeroDivisionError, "integer modulo by zero");
            return frac_make(int_mod(int_mul(n1, d2), int_mul(n2, d1)), int_mul(d1, d2));
        case Bin::Pow: {
            if (eq(d2, Value::integer(1))) {
                if (a.tag != Tag::Fraction && !int_is_neg(n2)) return number_binary(Bin::Pow, a, n2, false);
                if (n2.tag == Tag::Big) too_big();
                const int64_t p = small_of(n2);
                if (p >= 0) return frac_make(int_pow(n1, (uint64_t) p), int_pow(d1, (uint64_t) p));
                if (!int_is_neg(n1)) return frac_make(int_pow(d1, (uint64_t) -p), int_pow(n1, (uint64_t) -p));
                return frac_make(int_pow(int_negate(d1), (uint64_t) -p), int_pow(int_negate(n1), (uint64_t) -p));
            }
            return Value::real(float_pow(to_float(a), to_float(b)));
        }
        default: bad_operands(op, a, b, false);
    }
}
inline Value frac_from_text(const U32& raw) {
    std::size_t a = 0, b = raw.size();
    while (a < b && is_space(raw[a])) ++a;
    while (b > a && is_space(raw[b - 1])) --b;
    auto bad = [&]() { raise(T_ValueError, "Invalid literal for Fraction: " + repr_utf8(text(U32(raw)))); };
    std::size_t i = a;
    bool neg = false;
    if (i < b && (raw[i] == '+' || raw[i] == '-')) neg = raw[i++] == '-';
    auto digits = [&]() {
        std::string out;
        while (i < b && (ascii_digit(raw[i]) || (raw[i] == '_' && !out.empty() && i + 1 < b && ascii_digit(raw[i + 1])))) {
            if (raw[i] != '_') out.push_back((char) raw[i]);
            ++i;
        }
        return out;
    };
    const std::string whole = digits();
    Value num = Value::integer(0), den = Value::integer(1);
    if (i < b && raw[i] == '/') {
        ++i;
        const std::string under = digits();
        if (whole.empty() || under.empty() || i != b) bad();
        return frac_make(int_parse(whole, 10, neg), int_parse(under, 10, false));
    }
    std::string fraction;
    if (i < b && raw[i] == '.') {
        ++i;
        fraction = digits();
    }
    if (whole.empty() && fraction.empty()) bad();
    num = int_parse(whole + fraction, 10, false);
    int64_t exponent = -(int64_t) fraction.size();
    if (i < b && (raw[i] == 'e' || raw[i] == 'E')) {
        ++i;
        bool exp_neg = false;
        if (i < b && (raw[i] == '+' || raw[i] == '-')) exp_neg = raw[i++] == '-';
        const std::string e = digits();
        if (e.empty()) bad();
        const int64_t shift = (int64_t) std::min<std::size_t>(e.size(), 7) == (int64_t) e.size() ? std::stoll(e) : INT32_MAX;
        exponent += exp_neg ? -shift : shift;
    }
    if (i != b) bad();
    if (exponent > 0) num = int_mul(num, int_pow(Value::integer(10), (uint64_t) exponent));
    else if (exponent < 0) den = int_pow(Value::integer(10), (uint64_t) -exponent);
    return frac_make(neg ? int_negate(num) : num, den);
}
inline uint64_t hash_frac(const FractionObj* f) {
    const Value p = Value::integer((int64_t) kHashModulus);
    // den^(P-2) mod P, the inverse by Fermat; 0 when den is a multiple of P
    Value r = Value::integer(1), b = int_mod(f->den, p);
    for (uint64_t e = kHashModulus - 2; e; e >>= 1) {
        if (e & 1) r = int_mod(int_mul(r, b), p);
        b = int_mod(int_mul(b, b), p);
    }
    uint64_t h = 314159;
    if (!int_is_zero(r)) h = (uint64_t) int_mod(int_mul(int_mod(int_abs(f->num), p), r), p).i;
    return hash_mix(int_is_neg(f->num) ? -(int64_t) h : (int64_t) h);
}
inline Value frac_limit(const Value& self, const Value& max_den) {
    auto* f = self.as<FractionObj>();
    if (int_cmp(max_den, Value::integer(1)) < 0) raise(T_ValueError, "max_denominator should be at least 1");
    if (int_cmp(f->den, max_den) <= 0) return self;
    Value p0 = Value::integer(0), q0 = Value::integer(1), p1 = Value::integer(1), q1 = Value::integer(0);
    Value n = f->num, d = f->den;
    for (;;) {
        rt->tick();
        const Value a = int_floordiv(n, d);
        Value q2 = int_add(q0, int_mul(a, q1));
        if (int_cmp(q2, max_den) > 0) break;
        Value p2 = int_add(p0, int_mul(a, p1));
        p0 = std::move(p1);
        q0 = std::move(q1);
        p1 = std::move(p2);
        q1 = std::move(q2);
        Value rest = int_sub(n, int_mul(a, d));
        n = std::move(d);
        d = std::move(rest);
    }
    const Value k = int_floordiv(int_sub(max_den, q0), q1);
    const Value bound1 = frac_make(int_add(p0, int_mul(k, p1)), int_add(q0, int_mul(k, q1)));
    const Value bound2 = frac_make(p1, q1);
    auto distance = [&](const Value& x) {
        Value diff = frac_binary(Bin::Sub, x, self);
        return diff.tag == Tag::Fraction && int_is_neg(diff.as<FractionObj>()->num) ? frac_binary(Bin::Sub, Value::integer(0), diff) : diff;
    };
    return frac_cmp(distance(bound2), distance(bound1)) <= 0 ? bound2 : bound1;
}
inline Value f_limit_denominator(Call& c) {
    only_keywords(c, {"max_denominator"}, "limit_denominator");
    Value* kw = c.keyword("max_denominator");
    return frac_limit(c.a[0], kw ? *kw : c.n > 1 ? c.a[1] : Value::integer(1000000));
}
inline Value f_as_integer_ratio(Call& c) { return pair(frac_num(c.a[0]), frac_den(c.a[0])); }

inline Value n_bit_length(Call& c) { return Value::integer((int64_t) int_bits(c.a[0])); }
inline Value n_bit_count(Call& c) {
    const IntRef x(c.a[0]);
    int64_t n = 0;
    for (std::size_t i = 0; i < x.n; ++i) n += std::popcount(x.p[i]);
    return Value::integer(n);
}
inline Value n_conjugate(Call& c) { return c.a[0].tag == Tag::Bool ? Value::integer(small_of(c.a[0])) : c.a[0]; }
inline Value n_int_ratio(Call& c) { return pair(c.a[0].tag == Tag::Bool ? Value::integer(small_of(c.a[0])) : c.a[0], Value::integer(1)); }
inline Value n_is_integer(Call& c) { return Value::boolean(std::isfinite(c.a[0].d) && std::floor(c.a[0].d) == c.a[0].d); }
inline Value n_float_ratio(Call& c) {
    const Value f = frac_from_double(c.a[0].d);
    return pair(f.as<FractionObj>()->num, f.as<FractionObj>()->den);
}
inline Value r_count(Call& c) {
    return Value::integer(contains(c.a[0], c.a[1]) ? 1 : 0);
}
inline Value r_index(Call& c) {
    auto* r = c.a[0].as<RangeObj>();
    if (!is_int(c.a[1]) || !contains(c.a[0], c.a[1])) raise(T_ValueError, repr_utf8(c.a[1]) + " is not in range");
    return Value::integer((small_of(c.a[1]) - r->start) / r->step);
}
inline Value cache_clear(Call& c) {
    auto* f = c.a[0].as<CachedObj>();
    Table gone;
    std::swap(gone.entries, f->t.entries);
    std::swap(gone.index, f->t.index);
    f->t.used = 0;
    f->hits = f->misses = 0;
    return Value::none();
}

struct KnownMissing {
    int type;
    std::string_view names;
};
// In CPython, not here: the program gets a refusal, never an AttributeError CPython would not raise.
inline constexpr KnownMissing kKnownMissing[] = {
    {T_str, " encode format_map "},
    {T_int, " to_bytes from_bytes "},
    {T_float, " hex fromhex "},
    {T__lru_cache_wrapper, " cache_info cache_parameters "},
    {T_Fraction, " from_float from_decimal conjugate real imag "},
    {T_generator, " send throw close gi_code gi_frame gi_running gi_suspended gi_yieldfrom "},
    {T_slice, " indices "},
    {T_dict_keys, " isdisjoint mapping "},
    {T_dict_items, " isdisjoint mapping "},
    {T_dict_values, " mapping "},
    {T_BaseException, " add_note with_traceback "},
    {T_ImportError, " msg name path "},
    {T_StopIteration, " value "},
    {T_SyntaxError, " end_lineno end_offset filename lineno msg offset print_file_and_line text "},
    {T_OSError, " characters_written errno filename filename2 strerror winerror "},
    {T_NameError, " name "},
    {T_AttributeError, " name obj "},
    {T_SystemExit, " code "},
};
inline const NativeDef kStrMethods[] = {
    {"find", s_find, 2, 4}, {"rfind", s_rfind, 2, 4}, {"index", s_index, 2, 4}, {"rindex", s_rindex, 2, 4}, {"count", s_count, 2, 4},
    {"startswith", s_startswith, 2, 4}, {"endswith", s_endswith, 2, 4}, {"split", s_split, 1, 3, true}, {"rsplit", s_rsplit, 1, 3, true},
    {"splitlines", s_splitlines, 1, 2, true}, {"join", s_join, 2, 2}, {"replace", s_replace, 3, 4}, {"upper", s_upper, 1, 1},
    {"lower", s_lower, 1, 1}, {"casefold", s_casefold, 1, 1}, {"swapcase", s_swapcase, 1, 1}, {"capitalize", s_capitalize, 1, 1},
    {"title", s_title, 1, 1}, {"isdigit", s_isdigit, 1, 1}, {"isdecimal", s_isdecimal, 1, 1}, {"isnumeric", s_isnumeric, 1, 1},
    {"isalpha", s_isalpha, 1, 1}, {"isalnum", s_isalnum, 1, 1}, {"isspace", s_isspace, 1, 1}, {"isascii", s_isascii, 1, 1},
    {"isprintable", s_isprintable, 1, 1}, {"isidentifier", s_isidentifier, 1, 1}, {"islower", s_islower, 1, 1}, {"isupper", s_isupper, 1, 1},
    {"istitle", s_istitle, 1, 1}, {"strip", s_strip, 1, 2}, {"lstrip", s_lstrip, 1, 2}, {"rstrip", s_rstrip, 1, 2}, {"ljust", s_ljust, 2, 3},
    {"rjust", s_rjust, 2, 3}, {"center", s_center, 2, 3}, {"zfill", s_zfill, 2, 2}, {"expandtabs", s_expandtabs, 1, 2, true},
    {"partition", s_partition, 2, 2}, {"rpartition", s_rpartition, 2, 2}, {"removeprefix", s_removeprefix, 2, 2},
    {"removesuffix", s_removesuffix, 2, 2}, {"maketrans", s_maketrans, 1, 3, false, true}, {"translate", s_translate, 2, 2},
    {"format", s_format, 1, -1, true},
};
inline const NativeDef kListMethods[] = {
    {"append", l_append, 2, 2}, {"extend", l_extend, 2, 2}, {"clear", l_clear, 1, 1}, {"insert", l_insert, 3, 3}, {"pop", l_pop, 1, 2},
    {"remove", l_remove, 2, 2}, {"reverse", l_reverse, 1, 1}, {"copy", l_copy, 1, 1}, {"count", l_count, 2, 2}, {"index", l_index, 2, 4},
    {"sort", l_sort, 1, 1, true},
};
inline const NativeDef kTupleMethods[] = {{"count", t_count, 2, 2}, {"index", t_index, 2, 4}};
inline const NativeDef kDictMethods[] = {
    {"get", d_get, 2, 3}, {"keys", d_keys, 1, 1}, {"values", d_values, 1, 1}, {"items", d_items, 1, 1}, {"pop", d_pop, 2, 3},
    {"popitem", d_popitem, 1, 1}, {"setdefault", d_setdefault, 2, 3}, {"update", d_update, 1, 2, true}, {"clear", d_clear, 1, 1},
    {"copy", d_copy, 1, 1}, {"fromkeys", d_fromkeys, 1, 2, false, true},
};
inline const NativeDef kCounterMethods[] = {{"most_common", c_most_common, 1, 2}, {"elements", c_elements, 1, 1}, {"subtract", c_subtract, 1, 2}, {"total", c_total, 1, 1}, {"fromkeys", c_fromkeys, 0, -1, false, true}};
inline const NativeDef kSetMethods[] = {
    {"add", st_add, 2, 2}, {"discard", st_discard, 2, 2}, {"remove", st_remove, 2, 2}, {"pop", st_pop, 1, 1}, {"clear", st_clear, 1, 1},
    {"copy", st_copy, 1, 1}, {"union", st_union, 1, -1}, {"intersection", st_intersection, 1, -1}, {"difference", st_difference, 1, -1},
    {"symmetric_difference", st_symmetric_difference, 2, 2}, {"update", st_update, 1, -1}, {"intersection_update", st_intersection_update, 1, -1},
    {"difference_update", st_difference_update, 1, -1}, {"symmetric_difference_update", st_symmetric_difference_update, 2, 2},
    {"isdisjoint", st_isdisjoint, 2, 2}, {"issubset", st_issubset, 2, 2}, {"issuperset", st_issuperset, 2, 2},
};
inline const NativeDef kFrozensetMethods[] = {
    {"copy", st_copy, 1, 1}, {"union", st_union, 1, -1}, {"intersection", st_intersection, 1, -1}, {"difference", st_difference, 1, -1},
    {"symmetric_difference", st_symmetric_difference, 2, 2}, {"isdisjoint", st_isdisjoint, 2, 2}, {"issubset", st_issubset, 2, 2}, {"issuperset", st_issuperset, 2, 2},
};
inline const NativeDef kDequeMethods[] = {
    {"append", q_append, 2, 2}, {"appendleft", q_appendleft, 2, 2}, {"pop", q_pop, 1, 1}, {"popleft", q_popleft, 1, 1}, {"extend", q_extend, 2, 2},
    {"extendleft", q_extendleft, 2, 2}, {"clear", q_clear, 1, 1}, {"copy", q_copy, 1, 1}, {"count", q_count, 2, 2}, {"index", q_index, 2, 4},
    {"remove", q_remove, 2, 2}, {"insert", q_insert, 3, 3}, {"reverse", q_reverse, 1, 1}, {"rotate", q_rotate, 1, 2},
};
inline const NativeDef kIntMethods[] = {{"bit_length", n_bit_length, 1, 1}, {"bit_count", n_bit_count, 1, 1}, {"conjugate", n_conjugate, 1, 1}, {"as_integer_ratio", n_int_ratio, 1, 1}};
inline const NativeDef kFloatMethods[] = {{"is_integer", n_is_integer, 1, 1}, {"conjugate", n_conjugate, 1, 1}, {"as_integer_ratio", n_float_ratio, 1, 1}};
inline const NativeDef kRangeMethods[] = {{"count", r_count, 2, 2}, {"index", r_index, 2, 2}};
inline const NativeDef kFractionMethods[] = {{"limit_denominator", f_limit_denominator, 1, 2, true}, {"as_integer_ratio", f_as_integer_ratio, 1, 1}};
inline const NativeDef kCachedMethods[] = {{"cache_clear", cache_clear, 1, 1}};

using MethodMap = std::unordered_map<std::string_view, const NativeDef*>;
inline const NativeDef* method_for(int type, std::string_view name) {
    static const std::vector<MethodMap> tables = [] {
        std::vector<MethodMap> out((std::size_t) kTypeCount);
        auto add = [&](int t, const NativeDef* first, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) out[(std::size_t) t][first[i].name] = first + i;
        };
#define SANDBOX_ADD(t, table) add(t, table, sizeof(table) / sizeof(table[0]))
        SANDBOX_ADD(T_str, kStrMethods);
        SANDBOX_ADD(T_list, kListMethods);
        SANDBOX_ADD(T_tuple, kTupleMethods);
        SANDBOX_ADD(T_dict, kDictMethods);
        SANDBOX_ADD(T_Counter, kCounterMethods);
        SANDBOX_ADD(T_set, kSetMethods);
        SANDBOX_ADD(T_frozenset, kFrozensetMethods);
        SANDBOX_ADD(T_deque, kDequeMethods);
        SANDBOX_ADD(T_int, kIntMethods);
        SANDBOX_ADD(T_float, kFloatMethods);
        SANDBOX_ADD(T_range, kRangeMethods);
        SANDBOX_ADD(T_Fraction, kFractionMethods);
        SANDBOX_ADD(T__lru_cache_wrapper, kCachedMethods);
#undef SANDBOX_ADD
        return out;
    }();
    for (;; type = kTypes[type].base) {
        const auto& table = tables[(std::size_t) type];
        const auto hit = table.find(name);
        if (hit != table.end()) return hit->second;
        if (type == T_object) return nullptr;
    }
}
inline bool known_missing(int type, std::string_view name) {
    const std::string needle = " " + std::string(name) + " ";
    for (const auto& entry : kKnownMissing)
        if (entry.type == type && entry.names.find(needle) != std::string_view::npos) return true;
    return false;
}
inline Value module_attr(int module, std::string_view name);
inline const NativeDef* chain_from_iterable_def();
inline Value get_attr(const Value& obj, std::string_view name) {
    switch (obj.tag) {
        case Tag::Module: return module_attr((int) obj.i, name);
        case Tag::Class: {
            auto* c = obj.as<ClassObj>();
            if (name == "__name__") return text(c->name);
            if (const Value* m = find_attr(c->methods, name)) return *m;
            if (name.starts_with("__")) refuse(std::string(name) + " of a class");
            raise(T_AttributeError, "type object '" + std::string(c->name) + "' has no attribute '" + std::string(name) + "'");
        }
        case Tag::Instance: {
            auto* self = obj.as<InstanceObj>();
            if (const Value* a = find_attr(self->attrs, name)) return *a;
            if (const Value* m = find_attr(self->cls.as<ClassObj>()->methods, name)) return Value(new MethodObj(obj, *m));
            if (name.starts_with("__")) refuse(std::string(name) + " of an instance");
            raise(T_AttributeError, quoted(obj) + " object has no attribute '" + std::string(name) + "'");
        }
        case Tag::Type: {
            if (name == "__name__") {
                if (obj.i == T_iterator || obj.i == T_method) refuse("the type of a C-level iterator or method");
                return text(std::string_view(kTypes[obj.i].name));
            }
            if (obj.i == T_chain && name == "from_iterable") return Value::from(chain_from_iterable_def());
            if (const NativeDef* m = method_for((int) obj.i, name)) return Value::from(m);
            if (known_missing((int) obj.i, name)) refuse(std::string(kTypes[obj.i].name) + "." + std::string(name));
            raise(T_AttributeError, std::string("type object '") + kTypes[obj.i].name + "' has no attribute '" + std::string(name) + "'");
        }
        case Tag::Bool: case Tag::Int: case Tag::Big:
            if (name == "real" || name == "numerator") return obj.tag == Tag::Bool ? Value::integer(small_of(obj)) : obj;
            if (name == "imag") return Value::integer(0);
            if (name == "denominator") return Value::integer(1);
            break;
        case Tag::Float:
            if (name == "real") return obj;
            if (name == "imag") return Value::real(0.0);
            break;
        case Tag::Fraction:
            if (name == "numerator") return obj.as<FractionObj>()->num;
            if (name == "denominator") return obj.as<FractionObj>()->den;
            break;
        case Tag::Range:
            if (name == "start") return Value::integer(obj.as<RangeObj>()->start);
            if (name == "stop") return Value::integer(obj.as<RangeObj>()->stop);
            if (name == "step") return Value::integer(obj.as<RangeObj>()->step);
            break;
        case Tag::Slice:
            if (name == "start") return obj.as<SliceObj>()->start;
            if (name == "stop") return obj.as<SliceObj>()->stop;
            if (name == "step") return obj.as<SliceObj>()->step;
            break;
        case Tag::Dict:
            if (name == "default_factory" && obj.as<DictObj>()->kind == DictKind::Default) return obj.as<DictObj>()->factory;
            break;
        case Tag::Deque:
            if (name == "maxlen") return obj.as<DequeObj>()->maxlen < 0 ? Value::none() : Value::integer(obj.as<DequeObj>()->maxlen);
            break;
        case Tag::Exc:
            if (name == "args") {
                Vals args = obj.as<ExcObj>()->args;
                return make_tuple(std::move(args));
            }
            break;
        case Tag::Stub: return obj;
        default: break;
    }
    const int type = type_of(obj);
    if (const NativeDef* m = method_for(type, name)) {
        if (m->is_static) return Value::from(m);
        return Value(new BoundObj(obj, m));
    }
    for (int t = type;; t = kTypes[t].base) {
        if (known_missing(t, name)) refuse(std::string(kTypes[t].name) + "." + std::string(name));
        if (t == T_object) break;
    }
    if (name == "__name__" && (obj.tag == Tag::Native || obj.tag == Tag::Function || obj.tag == Tag::Cached)) refuse("__name__ of a function");
    raise(T_AttributeError, quoted(obj) + " object has no attribute '" + std::string(name) + "'");
}

inline void set_attr(const Value& obj, std::string_view name, Value v) {
    if (obj.tag != Tag::Instance) refuse(std::string("setting an attribute of a ") + type_name(obj));
    Attrs& attrs = obj.as<InstanceObj>()->attrs;
    for (auto& a : attrs) {
        if (a.first == name) {
            a.second = std::move(v);
            return;
        }
    }
    attrs.emplace_back(name, std::move(v));
}
inline void delete_attr(const Value& obj, std::string_view name) {
    if (obj.tag != Tag::Instance) refuse(std::string("deleting an attribute of a ") + type_name(obj));
    Attrs& attrs = obj.as<InstanceObj>()->attrs;
    for (std::size_t i = 0; i < attrs.size(); ++i) {
        if (attrs[i].first == name) {
            attrs.erase(attrs.begin() + (std::ptrdiff_t) i);
            return;
        }
    }
    raise(T_AttributeError, quoted(obj) + " object has no attribute '" + std::string(name) + "'");
}

enum class Tok : uint8_t { End, Newline, Indent, Dedent, Name, Keyword, Number, Text, Op, Error };
enum class Kw : uint8_t {
    False, None, True, And, As, Assert, Async, Await, Break, Class, Continue, Def, Del, Elif, Else, Except, Finally, For, From, Global, If,
    Import, In, Is, Lambda, Nonlocal, Not, Or, Pass, Raise, Return, Try, While, With, Yield
};
inline constexpr std::string_view kKeywords[] = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield"};
enum class P : uint8_t {
    LParen, RParen, LBracket, RBracket, LBrace, RBrace, Comma, Colon, Dot, Semi, Assign, Arrow, Walrus, Dots, Plus, Minus, Star, Power, Slash,
    FloorDiv, Percent, At, LShift, RShift, Amp, Bar, Caret, Tilde, Lt, Gt, Le, Ge, EqEq, Ne, PlusEq, MinusEq, StarEq, PowerEq, SlashEq,
    FloorDivEq, PercentEq, AtEq, AmpEq, BarEq, CaretEq, LShiftEq, RShiftEq
};
struct Token {
    Tok kind = Tok::End;
    uint8_t id = 0;
    int line = 0;
    uint32_t start = 0, end = 0;
};
using Tokens = std::vector<Token, Alloc<Token>>;
struct LexError {
    Status status = Status::Syntax;
    std::string text;
};
inline bool name_start(unsigned char c) { return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
inline bool name_char(unsigned char c) { return name_start(c) || (c >= '0' && c <= '9'); }

// CPython's tokenizer rules. `level` above zero reads the text of an f-string's `{...}`, where newlines mean nothing.
struct Lexer {
    std::string_view src;
    Tokens out;
    std::size_t pos, stop;
    int line, level;
    std::vector<int, Alloc<int>> indents{0}, alt_indents{0};
    std::vector<char, Alloc<char>> parens;
    LexError error;
    bool at_line_start = true;
    Lexer(std::string_view text, std::size_t begin, std::size_t end, int first_line, int nested)
        : src(text), pos(begin), stop(end), line(first_line), level(nested) {}
    char at(std::size_t i) const { return i < stop ? src[i] : '\0'; }
    void push(Tok kind, uint8_t id, std::size_t start, int token_line) {
        Token t;
        t.kind = kind;
        t.id = id;
        t.line = token_line;
        t.start = (uint32_t) start;
        t.end = (uint32_t) pos;
        out.push_back(t);
    }
    void fail(std::string text, Status status = Status::Syntax) {
        error.status = status;
        error.text = std::move(text);
        push(Tok::Error, 0, pos, line);
    }
    bool indentation() {
        int col = 0, alt = 0;
        for (;;) {
            const char c = at(pos);
            if (c == ' ') {
                ++col;
                ++alt;
            } else if (c == '\t') {
                col = (col / 8 + 1) * 8;
                alt = alt + 1;
            } else if (c == '\f') {
                col = alt = 0;
            } else if (c == '\\' && at(pos + 1) == '\n') {
                ++pos;
                ++line;
            } else {
                break;
            }
            ++pos;
        }
        const char c = at(pos);
        if (c == '#' || c == '\n' || pos >= stop) return true;
        if (col == indents.back()) {
            if (alt != alt_indents.back()) return tab_error();
        } else if (col > indents.back()) {
            if (alt <= alt_indents.back()) return tab_error();
            if (indents.size() >= 100) {
                fail("IndentationError: too many levels of indentation");
                return false;
            }
            indents.push_back(col);
            alt_indents.push_back(alt);
            push(Tok::Indent, 0, pos, line);
        } else {
            while (col < indents.back()) {
                indents.pop_back();
                alt_indents.pop_back();
                push(Tok::Dedent, 0, pos, line);
            }
            if (col != indents.back()) {
                fail("IndentationError: unindent does not match any outer indentation level");
                return false;
            }
            if (alt != alt_indents.back()) return tab_error();
        }
        return true;
    }
    bool tab_error() {
        fail("TabError: inconsistent use of tabs and spaces in indentation");
        return false;
    }
    bool digits(int (*valid)(char), bool leading_underscore_ok) {
        bool any = false;
        for (;;) {
            if (at(pos) == '_') {
                if (!leading_underscore_ok && !any) return false;
                if (!valid(at(pos + 1))) return false;
                ++pos;
            }
            if (!valid(at(pos))) return any;
            any = true;
            while (valid(at(pos))) ++pos;
        }
    }
    bool end_of_number() {
        const unsigned char c = (unsigned char) at(pos);
        if (!name_start(c) && c < 128) return true;
        for (const std::string_view word : {"and", "else", "for", "if", "in", "is", "not", "or"}) {
            if (src.substr(pos).substr(0, std::min(word.size(), stop - pos)) == word) return true;
        }
        fail("SyntaxError: invalid decimal literal");
        return false;
    }
    void number() {
        const std::size_t start = pos;
        auto dec = [](char x) -> int { return x >= '0' && x <= '9'; };
        auto hex = [](char x) -> int { return (x >= '0' && x <= '9') || (x >= 'a' && x <= 'f') || (x >= 'A' && x <= 'F'); };
        auto oct = [](char x) -> int { return x >= '0' && x <= '7'; };
        auto bin = [](char x) -> int { return x == '0' || x == '1'; };
        if (at(pos) == '0' && std::strchr("xXoObB", at(pos + 1)) && at(pos + 1)) {
            const char kind = (char) (at(pos + 1) | 32);
            pos += 2;
            const bool ok = digits(kind == 'x' ? +hex : kind == 'o' ? +oct : +bin, true);
            if (!ok || dec(at(pos))) {
                fail(std::string("SyntaxError: invalid ") + (kind == 'x' ? "hexadecimal" : kind == 'o' ? "octal" : "binary") + " literal");
                return;
            }
            if (!end_of_number()) return;
            push(Tok::Number, kind == 'x' ? 2 : kind == 'o' ? 3 : 4, start, line);
            return;
        }
        bool real = false;
        if (at(pos) != '.') {
            if (!digits(+dec, false)) {
                fail("SyntaxError: invalid decimal literal");
                return;
            }
        }
        if (at(pos) == '.') {
            real = true;
            ++pos;
            if (dec(at(pos)) && !digits(+dec, false)) {
                fail("SyntaxError: invalid decimal literal");
                return;
            }
        }
        if (at(pos) == 'e' || at(pos) == 'E') {
            std::size_t look = pos + 1;
            if (at(look) == '+' || at(look) == '-') ++look;
            if (dec(at(look))) {
                pos = look;
                digits(+dec, false);
                real = true;
            }
        }
        if (at(pos) == 'j' || at(pos) == 'J') {
            fail("unsupported: complex", Status::Unsupported);
            return;
        }
        if (!real && at(start) == '0') {
            for (std::size_t i = start; i < pos; ++i) {
                if (at(i) != '0' && at(i) != '_') {
                    fail("SyntaxError: leading zeros in decimal integer literals are not permitted");
                    return;
                }
            }
        }
        if (!end_of_number()) return;
        push(Tok::Number, real ? 1 : 0, start, line);
    }
    void string(std::size_t start) {
        const char quote = at(pos);
        const bool triple = at(pos + 1) == quote && at(pos + 2) == quote;
        const int first = line;
        pos += triple ? 3 : 1;
        for (;;) {
            if (pos >= stop) {
                fail(triple ? "SyntaxError: unterminated triple-quoted string literal" : "SyntaxError: unterminated string literal");
                return;
            }
            const char c = at(pos);
            if (c == '\\') {
                if (at(pos + 1) == '\n') ++line;
                pos += 2;
                continue;
            }
            if (c == '\n') {
                if (!triple) {
                    fail("SyntaxError: unterminated string literal");
                    return;
                }
                ++line;
            }
            if (c == quote && (!triple || (at(pos + 1) == quote && at(pos + 2) == quote))) {
                pos += triple ? 3 : 1;
                push(Tok::Text, 0, start, first);
                return;
            }
            ++pos;
        }
    }
    void name() {
        const std::size_t start = pos;
        while (name_char((unsigned char) at(pos))) ++pos;
        if ((unsigned char) at(pos) >= 128) {
            fail("unsupported: non-ASCII identifier", Status::Unsupported);
            return;
        }
        const char q = at(pos);
        if ((q == '\'' || q == '"') && pos - start <= 2) {
            std::string lower;
            for (std::size_t i = start; i < pos; ++i) lower.push_back((char) (at(i) | 32));
            if (lower == "r" || lower == "u" || lower == "b" || lower == "f" || lower == "br" || lower == "rb" || lower == "fr" || lower == "rf") {
                string(start);
                return;
            }
        }
        const std::string_view word = src.substr(start, pos - start);
        for (std::size_t k = 0; k < std::size(kKeywords); ++k) {
            if (kKeywords[k] == word) {
                push(Tok::Keyword, (uint8_t) k, start, line);
                return;
            }
        }
        push(Tok::Name, 0, start, line);
    }
    bool op() {
        const std::size_t start = pos;
        const char a = at(pos), b = at(pos + 1), c = at(pos + 2);
        auto emit = [&](P p, std::size_t width) {
            pos += width;
            push(Tok::Op, (uint8_t) p, start, line);
            return true;
        };
        auto open = [&](P p, char close) {
            parens.push_back(close);
            ++level;
            return emit(p, 1);
        };
        auto closing = [&](P p) {
            if (parens.empty()) {
                if (level > 0) return emit(p, 1);
                fail(std::string("SyntaxError: unmatched '") + a + "'");
                return false;
            }
            if (parens.back() != a) {
                fail(std::string("SyntaxError: closing parenthesis '") + a + "' does not match opening parenthesis");
                return false;
            }
            parens.pop_back();
            --level;
            return emit(p, 1);
        };
        switch (a) {
            case '(': return open(P::LParen, ')');
            case '[': return open(P::LBracket, ']');
            case '{': return open(P::LBrace, '}');
            case ')': return closing(P::RParen);
            case ']': return closing(P::RBracket);
            case '}': return closing(P::RBrace);
            case ',': return emit(P::Comma, 1);
            case ';': return emit(P::Semi, 1);
            case '~': return emit(P::Tilde, 1);
            case ':': return b == '=' ? emit(P::Walrus, 2) : emit(P::Colon, 1);
            case '.': return b == '.' && c == '.' ? emit(P::Dots, 3) : emit(P::Dot, 1);
            case '+': return b == '=' ? emit(P::PlusEq, 2) : emit(P::Plus, 1);
            case '-': return b == '=' ? emit(P::MinusEq, 2) : b == '>' ? emit(P::Arrow, 2) : emit(P::Minus, 1);
            case '*': return b == '*' ? (c == '=' ? emit(P::PowerEq, 3) : emit(P::Power, 2)) : b == '=' ? emit(P::StarEq, 2) : emit(P::Star, 1);
            case '/': return b == '/' ? (c == '=' ? emit(P::FloorDivEq, 3) : emit(P::FloorDiv, 2)) : b == '=' ? emit(P::SlashEq, 2) : emit(P::Slash, 1);
            case '%': return b == '=' ? emit(P::PercentEq, 2) : emit(P::Percent, 1);
            case '@': return b == '=' ? emit(P::AtEq, 2) : emit(P::At, 1);
            case '&': return b == '=' ? emit(P::AmpEq, 2) : emit(P::Amp, 1);
            case '|': return b == '=' ? emit(P::BarEq, 2) : emit(P::Bar, 1);
            case '^': return b == '=' ? emit(P::CaretEq, 2) : emit(P::Caret, 1);
            case '=': return b == '=' ? emit(P::EqEq, 2) : emit(P::Assign, 1);
            case '!':
                if (b == '=') return emit(P::Ne, 2);
                break;
            case '<': return b == '<' ? (c == '=' ? emit(P::LShiftEq, 3) : emit(P::LShift, 2)) : b == '=' ? emit(P::Le, 2) : emit(P::Lt, 1);
            case '>': return b == '>' ? (c == '=' ? emit(P::RShiftEq, 3) : emit(P::RShift, 2)) : b == '=' ? emit(P::Ge, 2) : emit(P::Gt, 1);
            default: break;
        }
        fail("SyntaxError: invalid syntax");
        return false;
    }
    void run() {
        while (true) {
            if (at_line_start && level == 0) {
                if (!indentation()) return;
                at_line_start = false;
            }
            while (at(pos) == ' ' || at(pos) == '\t' || at(pos) == '\f') ++pos;
            const char c = at(pos);
            if (pos >= stop) break;
            if (c == '#') {
                while (pos < stop && at(pos) != '\n') ++pos;
                continue;
            }
            if (c == '\n') {
                const bool blank = out.empty() || out.back().kind == Tok::Newline || out.back().kind == Tok::Indent || out.back().kind == Tok::Dedent;
                if (level == 0 && !blank) {
                    ++pos;
                    push(Tok::Newline, 0, pos - 1, line);
                } else {
                    ++pos;
                }
                ++line;
                at_line_start = level == 0;
                continue;
            }
            if (c == '\\') {
                if (at(pos + 1) == '\n') {
                    pos += 2;
                    ++line;
                    if (pos >= stop) {
                        fail("SyntaxError: unexpected EOF while parsing");
                        return;
                    }
                    continue;
                }
                fail("SyntaxError: unexpected character after line continuation character");
                return;
            }
            if (name_start((unsigned char) c)) name();
            else if ((c >= '0' && c <= '9') || (c == '.' && at(pos + 1) >= '0' && at(pos + 1) <= '9')) number();
            else if (c == '"' || c == '\'') string(pos);
            else if ((unsigned char) c >= 128) fail("unsupported: non-ASCII identifier", Status::Unsupported);
            else op();
            if (!out.empty() && out.back().kind == Tok::Error) return;
        }
        if (level == 0 || !parens.empty()) {
            if (!out.empty() && out.back().kind != Tok::Newline && out.back().kind != Tok::Dedent && out.back().kind != Tok::Indent && parens.empty()) push(Tok::Newline, 0, pos, line);
            while (indents.size() > 1) {
                indents.pop_back();
                push(Tok::Dedent, 0, pos, line);
            }
        }
        push(Tok::End, 0, pos, line);
    }
};

enum class K : uint8_t {
    Constant, Name, Binary, Unary, Not, BoolOp, Compare, IfExp, Lambda, Call, Attribute, Subscript, Slice, List, Tuple, Set, Dict, ListComp,
    SetComp, GenExp, DictComp, CompFor, Starred, Walrus, FString, Formatted, Keyword, Param,
    Block, ExprStmt, Assign, AugAssign, AnnAssign, If, While, For, Break, Continue, Pass, Def, Class, Return, Raise, Global, Nonlocal, Assert, Del,
    Import, ImportFrom, Alias, Try, Handler
};
struct Scope;
struct Node;
using NodeList = std::vector<Node*, Alloc<Node*>>;
using Names = std::vector<std::string_view, Alloc<std::string_view>>;
enum class CmpOp : uint8_t { Lt, Le, Gt, Ge, Eq, Ne, In, NotIn, Is, IsNot };
enum class ParamKind : uint8_t { Positional, Star, KeywordOnly, DoubleStar };
struct Node {
    K kind;
    uint8_t op = 0;
    int line = 0;
    int index = 0;           // constant index
    Node* a = nullptr;
    Node* b = nullptr;
    Node* c = nullptr;
    Node* d = nullptr;
    NodeList items, more;
    std::vector<uint8_t, Alloc<uint8_t>> ops;
    Names names;
    std::string_view name;
    Scope* scope = nullptr;
    Node(K k, int l) : kind(k), line(l) {}
};
struct Program {
    std::deque<Node, Alloc<Node>> nodes;
    Vals constants;
    std::deque<std::string, Alloc<std::string>> texts;   // names a dotted import builds
    int constant(Value v) {
        constants.push_back(std::move(v));
        return (int) constants.size() - 1;
    }
};

[[noreturn]] inline void syntax_error(int line, const std::string& message, const char* type = "SyntaxError") {
    throw Stop{Status::Syntax, std::string(type) + ": " + message + " (line " + std::to_string(line) + ")"};
}
[[noreturn]] inline void parse_refuse(int line, const std::string& what) {
    throw Stop{Status::Unsupported, "unsupported: " + what + " (line " + std::to_string(line) + ")"};
}
struct Parser {
    Program& program;
    std::string_view source;
    const Tokens* tokens;
    LexError* lex_error;
    std::size_t pos = 0;
    int depth = 0;
    Parser(Program& p, std::string_view s, const Tokens* t, LexError* e) : program(p), source(s), tokens(t), lex_error(e) {}

    struct Depth {
        Parser& parser;
        explicit Depth(Parser& p) : parser(p) {
            if (++parser.depth > rt->limits.parser_depth) {
                --parser.depth;
                throw Stop{Status::RecursionLimit, "RecursionError: parser nesting limit " + std::to_string(rt->limits.parser_depth) + " (line " + std::to_string(parser.cur().line) + ")"};
            }
            check_stack();
        }
        ~Depth() { --parser.depth; }
        Depth(const Depth&) = delete;
        Depth& operator=(const Depth&) = delete;
    };
    const Token& cur() const { return (*tokens)[pos]; }
    const Token& peek(std::size_t n = 1) const { return (*tokens)[std::min(pos + n, tokens->size() - 1)]; }
    int line() const { return cur().line; }
    std::string_view text_of(const Token& t) const { return source.substr(t.start, t.end - t.start); }
    void check_error() const {
        if (cur().kind != Tok::Error) return;
        if (lex_error->status == Status::Unsupported) throw Stop{Status::Unsupported, lex_error->text + " (line " + std::to_string(cur().line) + ")"};
        throw Stop{Status::Syntax, lex_error->text + " (line " + std::to_string(cur().line) + ")"};
    }
    void advance() {
        check_error();
        if (pos + 1 < tokens->size()) ++pos;
    }
    bool is_op(P p) const { return cur().kind == Tok::Op && cur().id == (uint8_t) p; }
    bool is_kw(Kw k) const { return cur().kind == Tok::Keyword && cur().id == (uint8_t) k; }
    bool accept(P p) {
        if (!is_op(p)) return false;
        advance();
        return true;
    }
    bool accept(Kw k) {
        if (!is_kw(k)) return false;
        advance();
        return true;
    }
    [[noreturn]] void invalid() const {
        check_error();
        syntax_error(line(), "invalid syntax");
    }
    void expect(P p) {
        if (!accept(p)) invalid();
    }
    void expect(Kw k) {
        if (!accept(k)) invalid();
    }
    Node* make(K kind, int at) { return &program.nodes.emplace_back(kind, at); }
    std::string_view name_token() {
        check_error();
        if (cur().kind != Tok::Name) invalid();
        const std::string_view out = text_of(cur());
        advance();
        return out;
    }

    Node* parse_file() {
        Node* block = make(K::Block, 1);
        while (cur().kind != Tok::End) {
            if (cur().kind == Tok::Newline) {
                advance();
                continue;
            }
            parse_statement(block->items);
        }
        check_error();
        return block;
    }
    void parse_statement(NodeList& into) {
        check_error();
        if (cur().kind == Tok::Indent) syntax_error(line(), "unexpected indent", "IndentationError");
        if (cur().kind == Tok::Keyword) {
            switch ((Kw) cur().id) {
                case Kw::If: into.push_back(parse_if()); return;
                case Kw::While: into.push_back(parse_while()); return;
                case Kw::For: into.push_back(parse_for()); return;
                case Kw::Try: into.push_back(parse_try()); return;
                case Kw::Def: into.push_back(parse_def(nullptr)); return;
                case Kw::Class:
                    if (peek().kind == Tok::Name) {
                        into.push_back(parse_class());
                        return;
                    }
                    invalid();
                case Kw::With:
                    if (peek().kind != Tok::Op || peek().id != (uint8_t) P::Assign) parse_refuse(line(), "with");
                    invalid();
                case Kw::Async:
                    if (peek().kind == Tok::Keyword) parse_refuse(line(), "async");
                    invalid();
                default: break;
            }
        }
        if (is_op(P::At)) {
            into.push_back(parse_decorated());
            return;
        }
        if (cur().kind == Tok::Name && text_of(cur()) == "match") refuse_match();
        parse_simple_line(into);
    }
    // `match` is a name until its line ends in a colon.
    void refuse_match() {
        const Token& next = peek();
        if (next.kind == Tok::Op && (next.id == (uint8_t) P::Assign || next.id == (uint8_t) P::Dot || next.id == (uint8_t) P::Comma)) return;
        int nested = 0;
        for (std::size_t i = pos + 1; i < tokens->size(); ++i) {
            const Token& t = (*tokens)[i];
            if (t.kind == Tok::Newline || t.kind == Tok::End || t.kind == Tok::Error) {
                const Token& last = (*tokens)[i - 1];
                if (last.kind == Tok::Op && last.id == (uint8_t) P::Colon && nested == 0) parse_refuse(line(), "match");
                return;
            }
            if (t.kind == Tok::Op && (t.id == (uint8_t) P::LParen || t.id == (uint8_t) P::LBracket || t.id == (uint8_t) P::LBrace)) ++nested;
            if (t.kind == Tok::Op && (t.id == (uint8_t) P::RParen || t.id == (uint8_t) P::RBracket || t.id == (uint8_t) P::RBrace)) --nested;
        }
    }
    void parse_simple_line(NodeList& into) {
        for (;;) {
            into.push_back(parse_simple());
            if (accept(P::Semi)) {
                if (cur().kind == Tok::Newline) break;
                continue;
            }
            break;
        }
        if (cur().kind == Tok::End) return;
        check_error();
        if (cur().kind != Tok::Newline) invalid();
        advance();
    }
    Node* parse_block(const char* what) {
        Depth guard(*this);
        expect(P::Colon);
        Node* block = make(K::Block, line());
        if (cur().kind != Tok::Newline) {
            parse_simple_line(block->items);
            return block;
        }
        const int at = line();
        advance();
        if (cur().kind != Tok::Indent) {
            check_error();
            syntax_error(at, std::string("expected an indented block after ") + what, "IndentationError");
        }
        advance();
        while (cur().kind != Tok::Dedent && cur().kind != Tok::End) {
            if (cur().kind == Tok::Newline) {
                advance();
                continue;
            }
            parse_statement(block->items);
        }
        check_error();
        if (cur().kind == Tok::Dedent) advance();
        return block;
    }
    Node* parse_if() {
        Node* node = make(K::If, line());
        advance();
        node->items.push_back(parse_named());
        node->items.push_back(parse_block("'if' statement"));
        for (;;) {
            if (accept(Kw::Elif)) {
                node->items.push_back(parse_named());
                node->items.push_back(parse_block("'elif' statement"));
            } else if (accept(Kw::Else)) {
                node->a = parse_block("'else' statement");
                break;
            } else {
                break;
            }
        }
        return node;
    }
    Node* parse_while() {
        Node* node = make(K::While, line());
        advance();
        node->a = parse_named();
        node->b = parse_block("'while' statement");
        if (accept(Kw::Else)) node->c = parse_block("'else' statement");
        return node;
    }
    Node* parse_for() {
        Node* node = make(K::For, line());
        advance();
        node->a = parse_target_list();
        expect(Kw::In);
        node->b = parse_star_list();
        node->c = parse_block("'for' statement");
        if (accept(Kw::Else)) node->d = parse_block("'else' statement");
        return node;
    }
    Node* parse_try() {
        Node* node = make(K::Try, line());
        advance();
        node->a = parse_block("'try' statement");
        while (is_kw(Kw::Except)) {
            Node* handler = make(K::Handler, line());
            advance();
            if (is_op(P::Star)) parse_refuse(line(), "except*");
            if (!is_op(P::Colon)) {
                handler->a = parse_expr();
                if (is_op(P::Comma)) syntax_error(line(), "multiple exception types must be parenthesized");
                if (accept(Kw::As)) handler->name = name_token();
            }
            handler->b = parse_block("'except' statement");
            node->items.push_back(handler);
        }
        if (is_kw(Kw::Else)) {
            if (node->items.empty()) invalid();
            advance();
            node->b = parse_block("'else' statement");
        }
        if (accept(Kw::Finally)) node->c = parse_block("'finally' statement");
        if (node->items.empty() && !node->c) syntax_error(line(), "expected 'except' or 'finally' block");
        return node;
    }
    Node* parse_decorated() {
        const int at = line();
        Node* decorator = nullptr;
        advance();
        decorator = parse_named();
        if (cur().kind != Tok::Newline) invalid();
        advance();
        const Node* head = decorator->kind == K::Call ? decorator->a : decorator;
        const std::string_view name = head->kind == K::Name || head->kind == K::Attribute ? head->name : std::string_view{};
        if (name != "lru_cache" && name != "cache") parse_refuse(at, "decorator");
        if (is_op(P::At)) parse_refuse(line(), "stacked decorators");
        if (is_kw(Kw::Class)) parse_refuse(line(), "a decorated class");
        if (is_kw(Kw::Async)) parse_refuse(line(), "async");
        if (!is_kw(Kw::Def)) invalid();
        return parse_def(decorator);
    }
    // Methods only. A base, a class attribute or a protocol method is refused here, never run with the wrong meaning.
    Node* parse_class() {
        Node* node = make(K::Class, line());
        advance();
        node->name = name_token();
        if (accept(P::LParen)) {
            if (!is_op(P::RParen)) {
                const Node* base = parse_named();
                if (base->kind != K::Name || base->name != "object" || !is_op(P::RParen)) parse_refuse(node->line, "a base class");
            }
            expect(P::RParen);
        }
        node->a = parse_block("class definition");
        for (const Node* item : node->a->items) {
            if (item->kind == K::Pass) continue;
            if (item->kind == K::ExprStmt && item->a->kind == K::Constant) continue;
            if (item->kind != K::Def) parse_refuse(item->line, "a class body with more than methods");
            if (item->name.starts_with("__") && item->name != "__init__") parse_refuse(item->line, std::string(item->name) + " in a class");
        }
        return node;
    }
    void parse_params(Node* owner, bool annotations, P closer) {
        bool seen_star = false, seen_double = false, seen_default = false;
        while (!is_op(closer)) {
            if (seen_double) syntax_error(line(), "arguments cannot follow var-keyword argument");
            const int at = line();
            if (is_op(P::Slash)) parse_refuse(at, "positional-only parameters");
            Node* param = make(K::Param, at);
            if (accept(P::Star)) {
                if (seen_star) syntax_error(at, "* argument may appear only once");
                seen_star = true;
                if (is_op(P::Comma)) {
                    if (peek().kind == Tok::Op && peek().id == (uint8_t) closer) syntax_error(at, "named arguments must follow bare *");
                    advance();
                    continue;
                }
                if (is_op(closer)) syntax_error(at, "named arguments must follow bare *");
                param->op = (uint8_t) ParamKind::Star;
            } else if (accept(P::Power)) {
                seen_double = true;
                param->op = (uint8_t) ParamKind::DoubleStar;
            } else {
                param->op = (uint8_t) (seen_star ? ParamKind::KeywordOnly : ParamKind::Positional);
            }
            param->name = name_token();
            if (annotations && accept(P::Colon)) parse_star_item();   // parsed and ignored
            if (accept(P::Assign)) {
                if ((ParamKind) param->op == ParamKind::Star || (ParamKind) param->op == ParamKind::DoubleStar) syntax_error(at, "var-positional argument cannot have default value");
                param->a = parse_expr();
                if ((ParamKind) param->op == ParamKind::Positional) seen_default = true;
            } else if ((ParamKind) param->op == ParamKind::Positional && seen_default) {
                syntax_error(at, "non-default argument follows default argument");
            }
            owner->items.push_back(param);
            if (!accept(P::Comma)) break;
        }
    }
    Node* parse_def(Node* decorator) {
        Node* node = make(K::Def, line());
        advance();
        node->name = name_token();
        expect(P::LParen);
        parse_params(node, true, P::RParen);
        expect(P::RParen);
        if (accept(P::Arrow)) parse_expr();
        node->c = decorator;
        node->a = parse_block("function definition");
        return node;
    }
    Node* parse_simple() {
        const int at = line();
        check_error();
        if (cur().kind == Tok::Keyword) {
            switch ((Kw) cur().id) {
                case Kw::Pass: advance(); return make(K::Pass, at);
                case Kw::Break: advance(); return make(K::Break, at);
                case Kw::Continue: advance(); return make(K::Continue, at);
                case Kw::Return: {
                    Node* node = make(K::Return, at);
                    advance();
                    if (cur().kind != Tok::Newline && !is_op(P::Semi) && cur().kind != Tok::End) node->a = parse_star_list();
                    return node;
                }
                case Kw::Raise: {
                    Node* node = make(K::Raise, at);
                    advance();
                    if (cur().kind != Tok::Newline && !is_op(P::Semi) && cur().kind != Tok::End) {
                        node->a = parse_expr();
                        if (accept(Kw::From)) node->b = parse_expr();
                    }
                    return node;
                }
                case Kw::Global: case Kw::Nonlocal: {
                    Node* node = make(cur().id == (uint8_t) Kw::Global ? K::Global : K::Nonlocal, at);
                    advance();
                    node->names.push_back(name_token());
                    while (accept(P::Comma)) node->names.push_back(name_token());
                    return node;
                }
                case Kw::Assert: {
                    Node* node = make(K::Assert, at);
                    advance();
                    node->a = parse_expr();
                    if (accept(P::Comma)) node->b = parse_expr();
                    return node;
                }
                case Kw::Del: {
                    Node* node = make(K::Del, at);
                    advance();
                    node->items.push_back(parse_postfix_target());
                    while (accept(P::Comma)) {
                        if (cur().kind == Tok::Newline || is_op(P::Semi) || cur().kind == Tok::End) break;
                        node->items.push_back(parse_postfix_target());
                    }
                    for (Node* target : node->items) check_delete_target(target);
                    return node;
                }
                case Kw::Import: return parse_import();
                case Kw::From: return parse_from();
                default: break;
            }
        }
        Node* first = parse_star_list();
        if (is_op(P::Assign)) {
            Node* node = make(K::Assign, at);
            check_target(first);
            node->items.push_back(first);
            Node* value = nullptr;
            while (accept(P::Assign)) {
                Node* next = parse_star_list();
                if (is_op(P::Assign)) {
                    check_target(next);
                    node->items.push_back(next);
                } else {
                    value = next;
                }
            }
            node->a = value;
            return node;
        }
        if (is_op(P::Colon) && (first->kind == K::Name || first->kind == K::Subscript || first->kind == K::Attribute)) {
            advance();
            Node* node = make(K::AnnAssign, at);
            parse_expr();
            node->a = first;
            if (accept(P::Assign)) node->b = parse_star_list();
            else if (first->kind != K::Name) parse_refuse(at, "an annotation with no value on a subscript");
            return node;
        }
        if (cur().kind == Tok::Op && cur().id >= (uint8_t) P::PlusEq) {
            if (first->kind != K::Name && first->kind != K::Subscript && first->kind != K::Attribute) syntax_error(at, "illegal expression for augmented assignment");
            Node* node = make(K::AugAssign, at);
            node->op = augmented_op((P) cur().id);
            advance();
            node->a = first;
            node->b = parse_star_list();
            return node;
        }
        Node* node = make(K::ExprStmt, at);
        node->a = first;
        return node;
    }
    static uint8_t augmented_op(P p) {
        switch (p) {
            case P::PlusEq: return (uint8_t) Bin::Add;
            case P::MinusEq: return (uint8_t) Bin::Sub;
            case P::StarEq: return (uint8_t) Bin::Mul;
            case P::SlashEq: return (uint8_t) Bin::Div;
            case P::FloorDivEq: return (uint8_t) Bin::FloorDiv;
            case P::PercentEq: return (uint8_t) Bin::Mod;
            case P::PowerEq: return (uint8_t) Bin::Pow;
            case P::LShiftEq: return (uint8_t) Bin::LShift;
            case P::RShiftEq: return (uint8_t) Bin::RShift;
            case P::AmpEq: return (uint8_t) Bin::And;
            case P::BarEq: return (uint8_t) Bin::Or;
            case P::CaretEq: return (uint8_t) Bin::Xor;
            default: return (uint8_t) Bin::MatMul;
        }
    }
    void check_target(Node* n) {
        switch (n->kind) {
            case K::Name: case K::Subscript: case K::Attribute: return;
            case K::Tuple: case K::List: {
                bool starred = false;
                for (Node* item : n->items) {
                    if (item->kind == K::Starred) {
                        if (starred) syntax_error(n->line, "multiple starred expressions in assignment");
                        starred = true;
                        check_target(item->a);
                    } else {
                        check_target(item);
                    }
                }
                return;
            }
            case K::Starred: syntax_error(n->line, "starred assignment target must be in a list or tuple");
            default: syntax_error(n->line, "cannot assign to expression");
        }
    }
    void check_delete_target(Node* n) {
        switch (n->kind) {
            case K::Name: case K::Subscript: case K::Attribute: return;
            case K::Tuple: case K::List:
                for (Node* item : n->items) check_delete_target(item);
                return;
            default: syntax_error(n->line, "cannot delete expression");
        }
    }
    Node* parse_postfix_target() { return parse_bitwise(0); }
    // A `for` target stops at `in`, which a full expression would read as a comparison.
    Node* parse_target_list() {
        const int at = line();
        Node* first = parse_target_item();
        if (is_op(P::Comma)) {
            Node* tuple = make(K::Tuple, at);
            tuple->items.push_back(first);
            while (accept(P::Comma)) {
                if (is_kw(Kw::In)) break;
                tuple->items.push_back(parse_target_item());
            }
            check_target(tuple);
            return tuple;
        }
        check_target(first);
        return first;
    }
    Node* parse_target_item() {
        if (is_op(P::Star)) {
            Node* star = make(K::Starred, line());
            advance();
            star->a = parse_bitwise(0);
            return star;
        }
        return parse_bitwise(0);
    }
    Node* parse_import() {
        Node* node = make(K::Import, line());
        advance();
        for (;;) {
            Node* alias = make(K::Alias, line());
            alias->name = dotted_name();
            if (accept(Kw::As)) alias->names.push_back(name_token());
            node->items.push_back(alias);
            if (!accept(P::Comma)) break;
        }
        return node;
    }
    std::string_view dotted_name() {
        std::string_view first = name_token();
        if (!is_op(P::Dot)) return first;
        std::string whole(first);
        while (accept(P::Dot)) {
            whole.push_back('.');
            whole += std::string(name_token());
        }
        return program.texts.emplace_back(std::move(whole));
    }
    Node* parse_from() {
        Node* node = make(K::ImportFrom, line());
        advance();
        if (is_op(P::Dot) || is_op(P::Dots)) parse_refuse(line(), "relative import");
        node->name = dotted_name();
        expect(Kw::Import);
        if (is_op(P::Star)) parse_refuse(line(), "import *");
        const bool grouped = accept(P::LParen);
        for (;;) {
            if (grouped && is_op(P::RParen)) break;
            Node* alias = make(K::Alias, line());
            alias->name = name_token();
            if (accept(Kw::As)) alias->names.push_back(name_token());
            node->items.push_back(alias);
            if (!accept(P::Comma)) break;
        }
        if (grouped) expect(P::RParen);
        return node;
    }

    bool ends_list() const {
        if (cur().kind == Tok::Newline || cur().kind == Tok::End) return true;
        if (cur().kind == Tok::Keyword) return cur().id == (uint8_t) Kw::For || cur().id == (uint8_t) Kw::In;
        if (cur().kind != Tok::Op) return false;
        const P p = (P) cur().id;
        return p == P::RParen || p == P::RBracket || p == P::RBrace || p == P::Assign || p == P::Colon || p == P::Semi || p >= P::PlusEq;
    }
    Node* parse_star_item() {
        if (is_op(P::Star)) {
            Node* star = make(K::Starred, line());
            advance();
            star->a = parse_bitwise(0);
            return star;
        }
        return parse_named();
    }
    Node* parse_star_list() {
        const int at = line();
        Node* first = parse_star_item();
        if (!is_op(P::Comma)) return first;
        Node* tuple = make(K::Tuple, at);
        tuple->items.push_back(first);
        while (accept(P::Comma)) {
            if (ends_list()) break;
            tuple->items.push_back(parse_star_item());
        }
        return tuple;
    }
    Node* parse_named() {
        if (cur().kind == Tok::Name && peek().kind == Tok::Op && peek().id == (uint8_t) P::Walrus) {
            Node* node = make(K::Walrus, line());
            Node* target = make(K::Name, line());
            target->name = text_of(cur());
            advance();
            advance();
            node->a = target;
            node->b = parse_expr();
            return node;
        }
        return parse_expr();
    }
    Node* parse_expr() {
        Depth guard(*this);
        if (is_kw(Kw::Lambda)) return parse_lambda();
        const int at = line();
        Node* body = parse_or();
        if (!is_kw(Kw::If)) return body;
        advance();
        Node* node = make(K::IfExp, at);
        node->b = body;
        node->a = parse_or();
        if (!accept(Kw::Else)) {
            check_error();
            syntax_error(line(), "expected 'else' after 'if' expression");
        }
        node->c = parse_expr();
        return node;
    }
    Node* parse_lambda() {
        Node* node = make(K::Lambda, line());
        advance();
        parse_params(node, false, P::Colon);
        expect(P::Colon);
        node->a = parse_expr();
        return node;
    }
    Node* parse_or() {
        const int at = line();
        Node* first = parse_and();
        if (!is_kw(Kw::Or)) return first;
        Node* node = make(K::BoolOp, at);
        node->op = 1;
        node->items.push_back(first);
        while (accept(Kw::Or)) node->items.push_back(parse_and());
        return node;
    }
    Node* parse_and() {
        const int at = line();
        Node* first = parse_not();
        if (!is_kw(Kw::And)) return first;
        Node* node = make(K::BoolOp, at);
        node->op = 0;
        node->items.push_back(first);
        while (accept(Kw::And)) node->items.push_back(parse_not());
        return node;
    }
    Node* parse_not() {
        if (is_kw(Kw::Not)) {
            Depth guard(*this);
            Node* node = make(K::Not, line());
            advance();
            node->a = parse_not();
            return node;
        }
        return parse_comparison();
    }
    bool compare_op(CmpOp& out) {
        if (cur().kind == Tok::Op) {
            switch ((P) cur().id) {
                case P::Lt: out = CmpOp::Lt; break;
                case P::Le: out = CmpOp::Le; break;
                case P::Gt: out = CmpOp::Gt; break;
                case P::Ge: out = CmpOp::Ge; break;
                case P::EqEq: out = CmpOp::Eq; break;
                case P::Ne: out = CmpOp::Ne; break;
                default: return false;
            }
            advance();
            return true;
        }
        if (is_kw(Kw::In)) {
            advance();
            out = CmpOp::In;
            return true;
        }
        if (is_kw(Kw::Not) && peek().kind == Tok::Keyword && peek().id == (uint8_t) Kw::In) {
            advance();
            advance();
            out = CmpOp::NotIn;
            return true;
        }
        if (is_kw(Kw::Is)) {
            advance();
            out = CmpOp::Is;
            if (accept(Kw::Not)) out = CmpOp::IsNot;
            return true;
        }
        return false;
    }
    Node* parse_comparison() {
        const int at = line();
        Node* first = parse_bitwise(0);
        CmpOp op = CmpOp::Lt;
        if (!compare_op(op)) return first;
        Node* node = make(K::Compare, at);
        node->a = first;
        do {
            node->ops.push_back((uint8_t) op);
            node->items.push_back(parse_bitwise(0));
        } while (compare_op(op));
        return node;
    }
    static int binary_level(P p) {
        switch (p) {
            case P::Bar: return 1;
            case P::Caret: return 2;
            case P::Amp: return 3;
            case P::LShift: case P::RShift: return 4;
            case P::Plus: case P::Minus: return 5;
            case P::Star: case P::Slash: case P::FloorDiv: case P::Percent: case P::At: return 6;
            default: return 0;
        }
    }
    static uint8_t binary_op(P p) {
        switch (p) {
            case P::Bar: return (uint8_t) Bin::Or;
            case P::Caret: return (uint8_t) Bin::Xor;
            case P::Amp: return (uint8_t) Bin::And;
            case P::LShift: return (uint8_t) Bin::LShift;
            case P::RShift: return (uint8_t) Bin::RShift;
            case P::Plus: return (uint8_t) Bin::Add;
            case P::Minus: return (uint8_t) Bin::Sub;
            case P::Star: return (uint8_t) Bin::Mul;
            case P::Slash: return (uint8_t) Bin::Div;
            case P::FloorDiv: return (uint8_t) Bin::FloorDiv;
            case P::Percent: return (uint8_t) Bin::Mod;
            default: return (uint8_t) Bin::MatMul;
        }
    }
    Node* parse_bitwise(int level) {
        Node* left = parse_unary();
        for (;;) {
            if (cur().kind != Tok::Op) return left;
            const P p = (P) cur().id;
            const int here = binary_level(p);
            if (here == 0 || here <= level) return left;
            const int at = line();
            advance();
            Node* node = make(K::Binary, at);
            node->op = binary_op(p);
            node->a = left;
            node->b = parse_bitwise(here);
            left = node;
        }
    }
    Node* parse_unary() {
        if (cur().kind == Tok::Op && (cur().id == (uint8_t) P::Minus || cur().id == (uint8_t) P::Plus || cur().id == (uint8_t) P::Tilde)) {
            Depth guard(*this);
            Node* node = make(K::Unary, line());
            node->op = cur().id;
            advance();
            node->a = parse_unary();
            return node;
        }
        return parse_power();
    }
    Node* parse_power() {
        const int at = line();
        if (is_kw(Kw::Await)) parse_refuse(at, "await");
        Node* base = parse_postfix();
        if (!is_op(P::Power)) return base;
        advance();
        Node* node = make(K::Binary, at);
        node->op = (uint8_t) Bin::Pow;
        node->a = base;
        node->b = parse_unary();
        return node;
    }
    Node* parse_postfix() {
        Node* node = parse_atom();
        for (;;) {
            const int at = line();
            if (is_op(P::Dot)) {
                advance();
                Node* attribute = make(K::Attribute, at);
                attribute->a = node;
                attribute->name = name_token();
                if (attribute->name.rfind("__", 0) == 0 && attribute->name != "__name__") parse_refuse(at, "dunder attribute access");
                node = attribute;
            } else if (is_op(P::LParen)) {
                node = parse_call(node);
            } else if (is_op(P::LBracket)) {
                advance();
                Node* sub = make(K::Subscript, at);
                sub->a = node;
                sub->b = parse_subscript_list();
                expect(P::RBracket);
                node = sub;
            } else {
                return node;
            }
        }
    }
    Node* parse_subscript_item() {
        const int at = line();
        Node* first = nullptr;
        if (!is_op(P::Colon)) {
            first = parse_star_item();
            if (!is_op(P::Colon)) return first;
        }
        Node* slice = make(K::Slice, at);
        slice->a = first;
        expect(P::Colon);
        if (!is_op(P::Colon) && !is_op(P::RBracket) && !is_op(P::Comma)) slice->b = parse_expr();
        if (accept(P::Colon)) {
            if (!is_op(P::RBracket) && !is_op(P::Comma)) slice->c = parse_expr();
        }
        return slice;
    }
    Node* parse_subscript_list() {
        const int at = line();
        if (is_op(P::RBracket)) invalid();
        Node* first = parse_subscript_item();
        if (!is_op(P::Comma)) return first;
        Node* tuple = make(K::Tuple, at);
        tuple->items.push_back(first);
        while (accept(P::Comma)) {
            if (is_op(P::RBracket)) break;
            tuple->items.push_back(parse_subscript_item());
        }
        return tuple;
    }
    Node* parse_call(Node* func) {
        Node* call = make(K::Call, line());
        call->a = func;
        advance();
        bool seen_keyword = false, seen_double = false;
        while (!is_op(P::RParen)) {
            const int at = line();
            if (accept(P::Star)) {
                if (seen_double) syntax_error(at, "iterable argument unpacking follows keyword argument unpacking");
                Node* star = make(K::Starred, at);
                star->a = parse_expr();
                call->items.push_back(star);
            } else if (accept(P::Power)) {
                seen_double = true;
                Node* keyword = make(K::Keyword, at);
                keyword->a = parse_expr();
                call->more.push_back(keyword);
            } else if (cur().kind == Tok::Name && peek().kind == Tok::Op && peek().id == (uint8_t) P::Assign) {
                Node* keyword = make(K::Keyword, at);
                keyword->name = text_of(cur());
                for (const Node* other : call->more)
                    if (other->name == keyword->name) syntax_error(at, "keyword argument repeated: " + std::string(keyword->name));
                advance();
                advance();
                keyword->a = parse_expr();
                call->more.push_back(keyword);
                seen_keyword = true;
            } else {
                if (cur().kind == Tok::Keyword && (cur().id == (uint8_t) Kw::True || cur().id == (uint8_t) Kw::False || cur().id == (uint8_t) Kw::None) && peek().kind == Tok::Op && peek().id == (uint8_t) P::Assign)
                    syntax_error(at, "cannot assign to " + std::string(text_of(cur())));
                Node* arg = parse_named();
                if (is_op(P::Assign)) syntax_error(at, "expression cannot contain assignment, perhaps you meant \"==\"?");
                if (is_kw(Kw::For) || is_kw(Kw::Async)) {
                    Node* gen = make(K::GenExp, at);
                    gen->a = arg;
                    parse_comp_clauses(gen);
                    if (!call->items.empty() || !call->more.empty() || !is_op(P::RParen)) syntax_error(at, "Generator expression must be parenthesized");
                    arg = gen;
                }
                if (seen_double) syntax_error(at, "positional argument follows keyword argument unpacking");
                if (seen_keyword) syntax_error(at, "positional argument follows keyword argument");
                call->items.push_back(arg);
            }
            if (!accept(P::Comma)) break;
        }
        expect(P::RParen);
        return call;
    }
    void parse_comp_clauses(Node* comp) {
        while (is_kw(Kw::For) || is_kw(Kw::Async)) {
            if (is_kw(Kw::Async)) parse_refuse(line(), "async");
            Node* clause = make(K::CompFor, line());
            advance();
            clause->a = parse_target_list();
            expect(Kw::In);
            clause->b = parse_or_ternary_free();
            while (accept(Kw::If)) clause->items.push_back(parse_or_ternary_free());
            comp->items.push_back(clause);
        }
    }
    // In a comprehension `if` starts a filter, so an unparenthesized ternary cannot appear here.
    Node* parse_or_ternary_free() {
        Depth guard(*this);
        if (is_kw(Kw::Lambda)) syntax_error(line(), "invalid syntax");
        return parse_or();
    }
    Node* parse_number() {
        const Token t = cur();
        const int at = t.line;
        std::string digits;
        for (const char c : text_of(t))
            if (c != '_') digits.push_back(c);
        Value v;
        if (t.id == 1) {
            v = Value::real(parse_float_ascii(digits));
        } else if (t.id == 0) {
            if (digits.size() > (std::size_t) rt->limits.integer_digits) syntax_error(at, digit_limit_message(digits.size(), true));
            v = int_parse(digits, 10, false);
        } else {
            v = int_parse(std::string_view(digits).substr(2), t.id == 2 ? 16 : t.id == 3 ? 8 : 2, false);
        }
        advance();
        Node* node = make(K::Constant, at);
        node->index = program.constant(std::move(v));
        return node;
    }
    Node* parse_atom() {
        const int at = line();
        check_error();
        switch (cur().kind) {
            case Tok::Name: {
                Node* node = make(K::Name, at);
                node->name = text_of(cur());
                advance();
                return node;
            }
            case Tok::Number: return parse_number();
            case Tok::Text: return parse_strings();
            case Tok::Keyword: {
                const Kw k = (Kw) cur().id;
                if (k == Kw::None || k == Kw::True || k == Kw::False) {
                    advance();
                    Node* node = make(K::Constant, at);
                    node->index = program.constant(k == Kw::None ? Value::none() : Value::boolean(k == Kw::True));
                    return node;
                }
                if (k == Kw::Yield) parse_refuse(at, "yield");
                if (k == Kw::Lambda) return parse_lambda();
                invalid();
            }
            case Tok::Op: break;
            default: invalid();
        }
        if (accept(P::Dots)) {
            Node* node = make(K::Constant, at);
            node->index = program.constant(Value::of(Tag::Ellipsis));
            return node;
        }
        if (accept(P::LParen)) return parse_paren(at);
        if (accept(P::LBracket)) return parse_list(at);
        if (accept(P::LBrace)) return parse_brace(at);
        invalid();
    }
    Node* parse_paren(int at) {
        if (accept(P::RParen)) return make(K::Tuple, at);
        if (is_kw(Kw::Yield)) parse_refuse(line(), "yield");
        Node* first = parse_star_item();
        if (is_kw(Kw::For) || is_kw(Kw::Async)) {
            Node* gen = make(K::GenExp, at);
            gen->a = first;
            parse_comp_clauses(gen);
            expect(P::RParen);
            return gen;
        }
        if (is_op(P::Comma)) {
            Node* tuple = make(K::Tuple, at);
            tuple->items.push_back(first);
            while (accept(P::Comma)) {
                if (is_op(P::RParen)) break;
                tuple->items.push_back(parse_star_item());
            }
            expect(P::RParen);
            return tuple;
        }
        expect(P::RParen);
        if (first->kind == K::Starred) syntax_error(at, "cannot use starred expression here");
        return first;
    }
    Node* parse_list(int at) {
        Node* list = make(K::List, at);
        if (accept(P::RBracket)) return list;
        Node* first = parse_star_item();
        if (is_kw(Kw::For) || is_kw(Kw::Async)) {
            Node* comp = make(K::ListComp, at);
            comp->a = first;
            parse_comp_clauses(comp);
            expect(P::RBracket);
            return comp;
        }
        list->items.push_back(first);
        while (accept(P::Comma)) {
            if (is_op(P::RBracket)) break;
            list->items.push_back(parse_star_item());
        }
        expect(P::RBracket);
        return list;
    }
    Node* parse_brace(int at) {
        if (accept(P::RBrace)) return make(K::Dict, at);
        if (accept(P::Power)) {
            Node* dict = make(K::Dict, at);
            dict->items.push_back(nullptr);
            dict->more.push_back(parse_bitwise(0));
            return parse_dict_rest(dict);
        }
        Node* first = parse_star_item();
        if (is_op(P::Colon)) {
            advance();
            Node* value = parse_expr();
            if (is_kw(Kw::For) || is_kw(Kw::Async)) {
                Node* comp = make(K::DictComp, at);
                comp->a = first;
                comp->b = value;
                parse_comp_clauses(comp);
                expect(P::RBrace);
                return comp;
            }
            Node* dict = make(K::Dict, at);
            dict->items.push_back(first);
            dict->more.push_back(value);
            return parse_dict_rest(dict);
        }
        if (is_kw(Kw::For) || is_kw(Kw::Async)) {
            Node* comp = make(K::SetComp, at);
            comp->a = first;
            parse_comp_clauses(comp);
            expect(P::RBrace);
            return comp;
        }
        Node* set = make(K::Set, at);
        set->items.push_back(first);
        while (accept(P::Comma)) {
            if (is_op(P::RBrace)) break;
            set->items.push_back(parse_star_item());
        }
        expect(P::RBrace);
        return set;
    }
    Node* parse_dict_rest(Node* dict) {
        while (accept(P::Comma)) {
            if (is_op(P::RBrace)) break;
            if (accept(P::Power)) {
                dict->items.push_back(nullptr);
                dict->more.push_back(parse_bitwise(0));
                continue;
            }
            Node* key = parse_expr();
            expect(P::Colon);
            dict->items.push_back(key);
            dict->more.push_back(parse_expr());
        }
        expect(P::RBrace);
        return dict;
    }

    // Literal text between the quotes, with its escapes read.
    static void read_escapes(std::string_view body, bool raw, int at, U32& out) {
        U32 decoded;
        if (!decode_utf8(body, decoded)) syntax_error(at, "(unicode error) invalid UTF-8");
        if (raw) {
            out += decoded;
            return;
        }
        for (std::size_t i = 0; i < decoded.size();) {
            const char32_t c = decoded[i];
            if (c != U'\\' || i + 1 >= decoded.size()) {
                out.push_back(c);
                ++i;
                continue;
            }
            const char32_t e = decoded[i + 1];
            i += 2;
            switch (e) {
                case U'\n': break;
                case U'\\': out.push_back(U'\\'); break;
                case U'\'': out.push_back(U'\''); break;
                case U'"': out.push_back(U'"'); break;
                case U'a': out.push_back(7); break;
                case U'b': out.push_back(8); break;
                case U'f': out.push_back(12); break;
                case U'n': out.push_back(10); break;
                case U'r': out.push_back(13); break;
                case U't': out.push_back(9); break;
                case U'v': out.push_back(11); break;
                case U'0': case U'1': case U'2': case U'3': case U'4': case U'5': case U'6': case U'7': {
                    char32_t v = e - U'0';
                    for (int k = 0; k < 2 && i < decoded.size() && decoded[i] >= U'0' && decoded[i] <= U'7'; ++k) v = v * 8 + (decoded[i++] - U'0');
                    out.push_back(v);
                    break;
                }
                case U'x': case U'u': case U'U': {
                    const int width = e == U'x' ? 2 : e == U'u' ? 4 : 8;
                    char32_t v = 0;
                    for (int k = 0; k < width; ++k) {
                        const char32_t h = i < decoded.size() ? decoded[i] : 0;
                        int d = -1;
                        if (h >= U'0' && h <= U'9') d = (int) (h - U'0');
                        else if (h >= U'a' && h <= U'f') d = (int) (h - U'a') + 10;
                        else if (h >= U'A' && h <= U'F') d = (int) (h - U'A') + 10;
                        if (d < 0) syntax_error(at, "(unicode error) truncated escape sequence");
                        v = v * 16 + (char32_t) d;
                        ++i;
                    }
                    if (v > 0x10FFFF) syntax_error(at, "(unicode error) illegal Unicode character");
                    out.push_back(v);
                    break;
                }
                case U'N': parse_refuse(at, "\\N{...} escapes");
                default:
                    out.push_back(U'\\');
                    out.push_back(e);
            }
        }
    }
    Node* constant_text(U32&& s, int at) {
        Node* node = make(K::Constant, at);
        node->index = program.constant(text(std::move(s)));
        return node;
    }
    // Reads `{{`, `}}` and each `{...}` from raw bytes, as far as `end`. `nesting` is above zero inside a format spec.
    void parse_fstring_body(std::size_t begin, std::size_t end, bool raw, int at, Node* out, int nesting) {
        U32 literal;
        std::size_t i = begin, run = begin;
        auto flush_run = [&](std::size_t stop) {
            if (stop > run) read_escapes(source.substr(run, stop - run), raw, at, literal);
        };
        auto flush_literal = [&]() {
            if (!literal.empty()) out->items.push_back(constant_text(std::move(literal), at));
            literal = U32{};
        };
        while (i < end) {
            const char c = source[i];
            if (c == '\\' && !raw && i + 1 < end && source[i + 1] != '{' && source[i + 1] != '}') {
                i += 2;
                continue;
            }
            if (c == '{' || c == '}') {
                if (i + 1 < end && source[i + 1] == c && nesting == 0) {
                    flush_run(i + 1);
                    i += 2;
                    run = i;
                    continue;
                }
                if (c == '}') syntax_error(at, "f-string: single '}' is not allowed");
                flush_run(i);
                flush_literal();
                i = parse_fstring_field(i + 1, end, raw, at, out, nesting);
                run = i;
                continue;
            }
            ++i;
        }
        flush_run(end);
        flush_literal();
    }
    std::size_t parse_fstring_field(std::size_t i, std::size_t end, bool raw, int at, Node* out, int nesting) {
        if (nesting >= 2) syntax_error(at, "f-string: expressions nested too deeply");
        const std::size_t begin = i;
        int level = 0;
        char quote = 0;
        std::size_t quote_width = 0;
        for (; i < end; ++i) {
            const char c = source[i];
            if (quote) {
                if (c == '\\') syntax_error(at, "f-string expression part cannot include a backslash");
                if (c == quote && (quote_width == 1 || (i + 2 < end && source[i + 1] == quote && source[i + 2] == quote))) {
                    i += quote_width - 1;
                    quote = 0;
                }
                continue;
            }
            if (c == '\\') syntax_error(at, "f-string expression part cannot include a backslash");
            if (c == '#') syntax_error(at, "f-string expression part cannot include '#'");
            if (c == '\'' || c == '"') {
                quote = c;
                quote_width = i + 2 < end && source[i + 1] == c && source[i + 2] == c ? 3 : 1;
                i += quote_width - 1;
                continue;
            }
            if (c == '(' || c == '[' || c == '{') {
                ++level;
                continue;
            }
            if (level > 0) {
                if (c == ')' || c == ']' || c == '}') --level;
                continue;
            }
            if (c == '}' || c == ':') break;
            if (c == '!' && i + 1 < end && source[i + 1] != '=') break;
            if (c == '=' || c == '!' || c == '<' || c == '>') {
                if (i + 1 < end && source[i + 1] == '=') {
                    ++i;
                    continue;
                }
                if (c == '=') break;
            }
        }
        if (quote) syntax_error(at, "f-string: unterminated string");
        if (i >= end) syntax_error(at, "f-string: expecting '}'");
        std::size_t expr_end = i;
        bool blank = true;
        for (std::size_t k = begin; k < expr_end; ++k) blank = blank && (source[k] == ' ' || source[k] == '\t' || source[k] == '\n');
        if (blank) syntax_error(at, "f-string: empty expression not allowed");
        Node* field = make(K::Formatted, at);
        field->a = parse_embedded(begin, expr_end, at);
        int conversion = 0;
        if (source[i] == '=') {
            ++i;
            while (i < end && (source[i] == ' ' || source[i] == '\t')) ++i;
            U32 literal;
            read_escapes(source.substr(begin, i - begin), true, at, literal);
            out->items.push_back(constant_text(std::move(literal), at));
            if (i < end && source[i] != ':' && source[i] != '!') conversion = 'r';
        }
        if (i < end && source[i] == '!') {
            ++i;
            if (i >= end || (source[i] != 'r' && source[i] != 's' && source[i] != 'a')) syntax_error(at, "f-string: invalid conversion character");
            conversion = source[i++];
        }
        if (i < end && source[i] == ':') {
            ++i;
            const std::size_t spec_begin = i;
            int open = 0;
            for (; i < end; ++i) {
                if (source[i] == '{') ++open;
                else if (source[i] == '}') {
                    if (open == 0) break;
                    --open;
                }
            }
            if (i >= end) syntax_error(at, "f-string: expecting '}'");
            Node* spec = make(K::FString, at);
            parse_fstring_body(spec_begin, i, raw, at, spec, nesting + 1);
            field->b = spec;
        }
        if (i >= end || source[i] != '}') syntax_error(at, "f-string: expecting '}'");
        field->op = (uint8_t) conversion;
        out->items.push_back(field);
        return i + 1;
    }
    Node* parse_embedded(std::size_t begin, std::size_t end, int at) {
        Depth guard(*this);
        Lexer lexer(source, begin, end, at, 1);
        lexer.run();
        Parser sub(program, source, &lexer.out, &lexer.error);
        sub.depth = depth;
        Node* expr = sub.parse_embedded_expr();
        return expr;
    }
    Node* parse_embedded_expr() {
        Node* expr = parse_star_list();
        check_error();
        if (cur().kind != Tok::End) invalid();
        return expr;
    }
    Node* parse_strings() {
        const int at = line();
        Node* parts = nullptr;
        U32 literal;
        while (cur().kind == Tok::Text) {
            const Token t = cur();
            const std::string_view whole = text_of(t);
            std::size_t quote = 0;
            bool raw = false, formatted = false;
            for (; whole[quote] != '\'' && whole[quote] != '"'; ++quote) {
                const char c = (char) (whole[quote] | 32);
                if (c == 'r') raw = true;
                else if (c == 'f') formatted = true;
                else if (c == 'b') parse_refuse(at, "bytes");
            }
            const bool triple = whole.size() >= quote + 6 && whole[quote + 1] == whole[quote] && whole[quote + 2] == whole[quote];
            const std::size_t width = triple ? 3 : 1;
            const std::size_t begin = t.start + quote + width, end = t.end - width;
            if (formatted) {
                if (!parts) parts = make(K::FString, at);
                if (!literal.empty()) parts->items.push_back(constant_text(std::move(literal), at));
                literal = U32{};
                parse_fstring_body(begin, end, raw, t.line, parts, 0);
            } else {
                read_escapes(source.substr(begin, end - begin), raw, t.line, literal);
            }
            advance();
        }
        check_error();
        if (!parts) return constant_text(std::move(literal), at);
        if (!literal.empty()) parts->items.push_back(constant_text(std::move(literal), at));
        return parts;
    }
};

enum class Op : uint16_t {
    LoadConst, LoadFast, StoreFast, DeleteFast, LoadDeref, StoreDeref, DeleteDeref, LoadClosure, LoadGlobal, StoreGlobal, DeleteGlobal,
    Pop, Dup, Dup2, Rot2, Rot3, Add, Sub, Mul, Binary, Not, Negate, Positive, Invert, Compare, Jump, JumpIfFalse, JumpIfTrue,
    JumpIfFalseOrPop, JumpIfTrueOrPop, GetIter, ForIter, BuildList, BuildTuple, BuildSet, BuildDict, BuildSlice, BuildString, FormatValue,
    ListAppend, SetAdd, MapAdd, ListExtend, SetUpdate, DictUpdate, DictMerge, ListToTuple, Subscr, StoreSubscr, DeleteSubscr, LoadAttr, KwNames,
    Call, CallMethod, CallEx, MakeFunction, Return, Yield, Unpack, UnpackEx, Raise, Reraise, SetupTry, PopBlock, PopExcept, PushHandled,
    ExcMatch, Import, ImportFrom, BuildClass, AddMethod, StoreAttr, DeleteAttr
};
struct Instr {
    Op op;
    uint16_t b;
    int32_t a;
};
struct Code {
    std::vector<Instr, Alloc<Instr>> ins;
    std::vector<int, Alloc<int>> lines;
    Names names;      // of attributes and imports
    Names slots;      // of locals, then free variables
    std::vector<int, Alloc<int>> cells;   // the local slots read through a cell
    std::string qualname;
    int argcount = 0, kwonly = 0, star = -1, double_star = -1;
    int nlocals = 0, nfree = 0, nslots = 0, max_stack = 0;
    bool generator = false;
    int frame_size() const { return nslots + max_stack + 1; }
};
inline constexpr int kBound = 1, kUsed = 2, kDeclGlobal = 4, kDeclNonlocal = 8, kParam = 16, kCell = 32, kFree = 64;
using SymbolMap = std::map<std::string_view, int, std::less<>, Alloc<std::pair<const std::string_view, int>>>;
struct Scope {
    enum class Kind : uint8_t { Module, Function, Comprehension } kind;
    Scope* parent;
    SymbolMap symbols;
    std::vector<Scope*, Alloc<Scope*>> children;
    Scope(Kind k, Scope* p) : kind(k), parent(p) {}
};
// Names CPython has and this interpreter must not offer. Refused at compile time unless the program defines the name itself.
inline constexpr std::string_view kRefusedNames =
    " eval exec open input __import__ compile globals locals vars dir getattr setattr delattr hasattr id breakpoint help memoryview "
    "exit quit super property staticmethod classmethod bytes bytearray complex __file__ __builtins__ __doc__ __spec__ __loader__ "
    "__package__ __annotations__ __cached__ __debug__ license copyright credits NotImplemented ";

struct ScopeBuilder {
    std::deque<Scope, Alloc<Scope>>& scopes;
    Scope* module = nullptr;
    Scope* open(Scope::Kind kind, Scope* parent) {
        Scope* s = &scopes.emplace_back(kind, parent);
        if (parent) parent->children.push_back(s);
        return s;
    }
    void use(Scope* s, std::string_view name) { s->symbols[name] |= kUsed; }
    void bind(Scope* s, std::string_view name) {
        int& flags = s->symbols[name];
        if (flags & kDeclGlobal) module->symbols[name] |= kBound;
        flags |= kBound;
    }
    void bind_target(Scope* s, Node* t) {
        switch (t->kind) {
            case K::Name: bind(s, t->name); return;
            case K::Tuple: case K::List:
                for (Node* item : t->items) bind_target(s, item);
                return;
            case K::Starred: bind_target(s, t->a); return;
            default: walk(s, t);
        }
    }
    void declare(Scope* s, Node* n) {
        const bool global = n->kind == K::Global;
        const char* word = global ? "global" : "nonlocal";
        if (!global && s->kind == Scope::Kind::Module) syntax_error(n->line, "nonlocal declaration not allowed at module level");
        for (const auto name : n->names) {
            int& flags = s->symbols[name];
            const std::string quoted_name = "name '" + std::string(name) + "'";
            if (flags & kParam) syntax_error(n->line, quoted_name + " is parameter and " + word);
            if (global && (flags & kDeclNonlocal)) syntax_error(n->line, quoted_name + " is nonlocal and global");
            if (!global && (flags & kDeclGlobal)) syntax_error(n->line, quoted_name + " is nonlocal and global");
            if (s->kind != Scope::Kind::Module) {
                if (flags & kBound) syntax_error(n->line, quoted_name + " is assigned to before " + word + " declaration");
                if (flags & kUsed) syntax_error(n->line, quoted_name + " is used prior to " + word + " declaration");
            }
            flags |= global ? kDeclGlobal : kDeclNonlocal;
            if (global) module->symbols[name] |= kBound;
        }
    }
    void function(Scope* s, Node* n) {
        for (Node* p : n->items)
            if (p->a) walk(s, p->a);
        Scope* inner = open(Scope::Kind::Function, s);
        n->scope = inner;
        for (Node* p : n->items) {
            int& flags = inner->symbols[p->name];
            if (flags & kParam) syntax_error(n->line, "duplicate argument '" + std::string(p->name) + "' in function definition");
            flags |= kParam | kBound;
        }
        walk(inner, n->a);
    }
    void comprehension(Scope* s, Node* n) {
        walk(s, n->items[0]->b);
        Scope* inner = open(Scope::Kind::Comprehension, s);
        n->scope = inner;
        inner->symbols[".0"] = kParam | kBound;
        for (std::size_t i = 0; i < n->items.size(); ++i) {
            Node* clause = n->items[i];
            if (i) walk(inner, clause->b);
            bind_target(inner, clause->a);
            for (Node* test : clause->items) walk(inner, test);
        }
        walk(inner, n->a);
        if (n->b) walk(inner, n->b);
    }
    void walk_list(Scope* s, const NodeList& list) {
        for (Node* n : list)
            if (n) walk(s, n);
    }
    void walk(Scope* s, Node* n) {
        check_stack();
        switch (n->kind) {
            case K::Name: use(s, n->name); return;
            case K::Constant: case K::Pass: case K::Break: case K::Continue: return;
            case K::Lambda: function(s, n); return;
            case K::ListComp: case K::SetComp: case K::GenExp: case K::DictComp: comprehension(s, n); return;
            case K::Walrus:
                if (s->kind == Scope::Kind::Comprehension) parse_refuse(n->line, "walrus inside a comprehension");
                walk(s, n->b);
                bind(s, n->a->name);
                return;
            case K::Def:
                if (n->c) walk(s, n->c);
                bind(s, n->name);
                function(s, n);
                return;
            case K::Class:   // no scope of its own: a method sees what surrounds the class, and its name binds nowhere
                bind(s, n->name);
                for (Node* item : n->a->items) {
                    if (item->kind != K::Def) continue;
                    if (item->c) walk(s, item->c);
                    function(s, item);
                }
                return;
            case K::Assign:
                walk(s, n->a);
                for (Node* t : n->items) bind_target(s, t);
                return;
            case K::AugAssign:
                if (n->a->kind == K::Name) {
                    use(s, n->a->name);
                    bind(s, n->a->name);
                } else {
                    walk(s, n->a);
                }
                walk(s, n->b);
                return;
            case K::AnnAssign:
                if (n->b) walk(s, n->b);
                bind_target(s, n->a);
                return;
            case K::For:
                walk(s, n->b);
                bind_target(s, n->a);
                walk(s, n->c);
                if (n->d) walk(s, n->d);
                return;
            case K::Return:
                if (s->kind == Scope::Kind::Module) syntax_error(n->line, "'return' outside function");
                if (n->a) walk(s, n->a);
                return;
            case K::Global: case K::Nonlocal: declare(s, n); return;
            case K::Del:
                for (Node* t : n->items) bind_target(s, t);
                return;
            case K::Import:
                for (Node* alias : n->items) {
                    const std::string_view whole = alias->name;
                    bind(s, alias->names.empty() ? whole.substr(0, whole.find('.')) : alias->names[0]);
                }
                return;
            case K::ImportFrom:
                for (Node* alias : n->items) bind(s, alias->names.empty() ? alias->name : alias->names[0]);
                return;
            case K::Handler:
                if (n->a) walk(s, n->a);
                if (!n->name.empty()) bind(s, n->name);
                walk(s, n->b);
                return;
            default:
                if (n->a) walk(s, n->a);
                if (n->b) walk(s, n->b);
                if (n->c) walk(s, n->c);
                if (n->d) walk(s, n->d);
                walk_list(s, n->items);
                walk_list(s, n->more);
        }
    }
    void capture(Scope* s, std::string_view name, int flags, int line) {
        if (flags & kDeclGlobal) return;
        if ((flags & kBound) && !(flags & kDeclNonlocal)) return;
        if (!(flags & (kUsed | kDeclNonlocal))) return;
        for (Scope* up = s->parent; up && up->kind != Scope::Kind::Module; up = up->parent) {
            const auto hit = up->symbols.find(name);
            if (hit == up->symbols.end()) continue;
            if (hit->second & kDeclGlobal) break;
            if ((hit->second & kBound) && !(hit->second & kDeclNonlocal)) {
                hit->second |= kCell;
                for (Scope* between = s; between != up; between = between->parent) between->symbols[name] |= kFree;
                return;
            }
        }
        if (flags & kDeclNonlocal) syntax_error(line, "no binding for nonlocal '" + std::string(name) + "' found");
    }
    void resolve(Scope* s, int line) {
        check_stack();
        if (s->kind != Scope::Kind::Module) {
            std::vector<std::pair<std::string_view, int>, Alloc<std::pair<std::string_view, int>>> names(s->symbols.begin(), s->symbols.end());
            for (const auto& [name, flags] : names) capture(s, name, flags, line);
        }
        for (Scope* child : s->children) resolve(child, line);
    }
};

struct Compiler {
    Program& program;
    std::deque<Scope, Alloc<Scope>> scopes;
    std::deque<Code, Alloc<Code>> codes;
    std::map<std::string_view, int, std::less<>, Alloc<std::pair<const std::string_view, int>>> global_index;
    Names global_names;
    Scope* module_scope = nullptr;
    int empty_constant = -1;
    explicit Compiler(Program& p) : program(p) {}

    enum class CtxKind : uint8_t { Loop, Try, Finally, Handler, AsName, FinallyHandler };
    struct Ctx {
        CtxKind kind;
        bool for_loop = false;
        int break_label = 0, continue_label = 0;
        Node* body = nullptr;
        std::string_view name;
    };
    struct Unit {
        Code* code;
        Scope* scope;
        int depth = 0, line = 0, ret_slot = -1;
        std::vector<Ctx, Alloc<Ctx>> ctx;
        std::vector<int, Alloc<int>> labels, label_depth;
        std::vector<std::pair<int, int>, Alloc<std::pair<int, int>>> patches;
    };
    Unit* u = nullptr;

    int stack_effect(Op op, int a, int b) const {
        switch (op) {
            case Op::LoadConst: case Op::LoadFast: case Op::LoadDeref: case Op::LoadClosure: case Op::LoadGlobal: case Op::Dup: case Op::Import: case Op::ImportFrom: return 1;
            case Op::Dup2: return 2;
            case Op::StoreFast: case Op::StoreDeref: case Op::StoreGlobal: case Op::Pop: case Op::Add: case Op::Sub: case Op::Mul: case Op::Binary: case Op::Compare:
            case Op::JumpIfFalse: case Op::JumpIfTrue: case Op::JumpIfFalseOrPop: case Op::JumpIfTrueOrPop: case Op::ListAppend: case Op::SetAdd: case Op::ListExtend:
            case Op::SetUpdate: case Op::DictUpdate: case Op::DictMerge: case Op::Subscr: case Op::Return: case Op::Yield: case Op::Reraise: case Op::PopBlock: case Op::ExcMatch: return -1;
            case Op::BuildClass: return 1;
            case Op::AddMethod: case Op::DeleteAttr: return -1;
            case Op::MapAdd: case Op::DeleteSubscr: case Op::BuildSlice: case Op::StoreAttr: return -2;
            case Op::StoreSubscr: return -3;
            case Op::ForIter: case Op::SetupTry: return 1;
            case Op::BuildList: case Op::BuildTuple: case Op::BuildSet: case Op::BuildString: return 1 - a;
            case Op::BuildDict: return 1 - 2 * a;
            case Op::FormatValue: return b ? -1 : 0;
            case Op::Call: return -a;
            case Op::CallMethod: return -b;
            case Op::CallEx: return b ? -2 : -1;
            case Op::MakeFunction: return 1 - ((b & 1) + ((b >> 1) & 1) + ((b >> 2) & 1));
            case Op::Unpack: return a - 1;
            case Op::UnpackEx: return a + b;
            case Op::Raise: return -a;
            default: return 0;
        }
    }
    int emit(Op op, int a = 0, int b = 0) {
        Code* code = u->code;
        Instr in{op, (uint16_t) b, a};
        if (b > 65535) parse_refuse(u->line, "more than 65535 arguments in one call");
        code->ins.push_back(in);
        code->lines.push_back(u->line);
        u->depth += stack_effect(op, a, b);
        if (u->depth > code->max_stack) code->max_stack = u->depth;
        return (int) code->ins.size() - 1;
    }
    int new_label() {
        u->labels.push_back(-1);
        u->label_depth.push_back(-1);
        return (int) u->labels.size() - 1;
    }
    void bind(int label) {
        u->labels[(std::size_t) label] = (int) u->code->ins.size();
        if (u->label_depth[(std::size_t) label] >= 0) u->depth = u->label_depth[(std::size_t) label];
    }
    void jump(Op op, int label) {
        int depth_at_target = u->depth;
        if (op == Op::JumpIfFalse || op == Op::JumpIfTrue) depth_at_target -= 1;
        if (op == Op::ForIter) depth_at_target -= 1;
        const int at = emit(op);
        u->label_depth[(std::size_t) label] = depth_at_target;
        u->patches.emplace_back(at, label);
    }
    void finish(Unit& unit) {
        for (const auto& [at, label] : unit.patches) unit.code->ins[(std::size_t) at].a = unit.labels[(std::size_t) label];
    }

    int global_slot(std::string_view name) {
        const auto hit = global_index.find(name);
        if (hit != global_index.end()) return hit->second;
        const int slot = (int) global_names.size();
        global_names.push_back(name);
        global_index.emplace(name, slot);
        return slot;
    }
    int name_slot(std::string_view name) const {
        const auto& slots = u->code->slots;
        for (std::size_t i = 0; i < slots.size(); ++i)
            if (slots[i] == name) return (int) i;
        return -1;
    }
    enum class Access : uint8_t { Fast, Deref, Global };
    Access access(std::string_view name) const {
        if (u->scope->kind == Scope::Kind::Module) return Access::Global;
        const auto hit = u->scope->symbols.find(name);
        if (hit == u->scope->symbols.end()) return Access::Global;
        const int flags = hit->second;
        if (flags & kDeclGlobal) return Access::Global;
        if (flags & (kFree | kCell)) return Access::Deref;
        if (flags & kBound) return Access::Fast;
        return Access::Global;
    }
    void load_name(std::string_view name, int line) {
        switch (access(name)) {
            case Access::Fast: emit(Op::LoadFast, name_slot(name)); return;
            case Access::Deref: emit(Op::LoadDeref, name_slot(name)); return;
            default: break;
        }
        const auto hit = module_scope->symbols.find(name);
        const bool defined = hit != module_scope->symbols.end() && (hit->second & kBound);
        if (!defined && name != "__name__" && kRefusedNames.find(" " + std::string(name) + " ") != std::string_view::npos) parse_refuse(line, std::string(name));
        emit(Op::LoadGlobal, global_slot(name));
    }
    void store_name(std::string_view name) {
        switch (access(name)) {
            case Access::Fast: emit(Op::StoreFast, name_slot(name)); return;
            case Access::Deref: emit(Op::StoreDeref, name_slot(name)); return;
            default: emit(Op::StoreGlobal, global_slot(name));
        }
    }
    void delete_name(std::string_view name) {
        switch (access(name)) {
            case Access::Fast: emit(Op::DeleteFast, name_slot(name)); return;
            case Access::Deref: emit(Op::DeleteDeref, name_slot(name)); return;
            default: emit(Op::DeleteGlobal, global_slot(name));
        }
    }
    int attr_index(std::string_view name) {
        auto& names = u->code->names;
        for (std::size_t i = 0; i < names.size(); ++i)
            if (names[i] == name) return (int) i;
        names.push_back(name);
        return (int) names.size() - 1;
    }
    void load_none() { emit(Op::LoadConst, none_constant()); }
    int none_index = -1;
    int none_constant() {
        if (none_index < 0) none_index = program.constant(Value::none());
        return none_index;
    }

    // Lays out a function's frame: parameters, then other locals, then free variables.
    void prepare_slots(Code* code, Scope* scope, const NodeList* params) {
        if (params) {
            for (int pass = 0; pass < 2; ++pass) {
                for (const Node* p : *params) {
                    const ParamKind kind = (ParamKind) p->op;
                    if (pass == 0 ? kind == ParamKind::Positional : kind == ParamKind::KeywordOnly) code->slots.push_back(p->name);
                    if (kind == ParamKind::Positional && pass == 0) ++code->argcount;
                    if (kind == ParamKind::KeywordOnly && pass == 1) ++code->kwonly;
                }
            }
            for (const Node* p : *params) {
                if ((ParamKind) p->op == ParamKind::Star) {
                    code->star = (int) code->slots.size();
                    code->slots.push_back(p->name);
                }
            }
            for (const Node* p : *params) {
                if ((ParamKind) p->op == ParamKind::DoubleStar) {
                    code->double_star = (int) code->slots.size();
                    code->slots.push_back(p->name);
                }
            }
        }
        for (const auto& [name, flags] : scope->symbols) {
            if ((flags & kParam) || (flags & (kDeclGlobal | kDeclNonlocal)) || !(flags & kBound)) continue;
            if (flags & kFree) continue;
            code->slots.push_back(name);
        }
        for (std::size_t i = 0; i < code->slots.size(); ++i) {
            const auto hit = scope->symbols.find(code->slots[i]);
            if (hit != scope->symbols.end() && (hit->second & kCell)) code->cells.push_back((int) i);
        }
        code->nlocals = (int) code->slots.size();
        for (const auto& [name, flags] : scope->symbols) {
            if (flags & kFree) {
                code->slots.push_back(name);
                ++code->nfree;
            }
        }
        code->nslots = (int) code->slots.size();
    }
    Code* new_code(std::string qualname) {
        Code* code = &codes.emplace_back();
        code->qualname = std::move(qualname);
        return code;
    }
    int code_index(const Code* code) const {
        for (std::size_t i = codes.size(); i-- > 0;)
            if (&codes[i] == code) return (int) i;
        return -1;
    }
    std::string child_qualname(std::string_view name) const {
        if (!u || u->scope->kind == Scope::Kind::Module) return std::string(name);
        return u->code->qualname + ".<locals>." + std::string(name);
    }
    void make_closure_and_function(Code* code, Scope* scope, int flags) {
        int closure = 0;
        for (const auto& [name, symbol] : scope->symbols) {
            if (symbol & kFree) {
                emit(Op::LoadClosure, name_slot(name));
                ++closure;
            }
        }
        if (closure) {
            emit(Op::BuildTuple, closure);
            flags |= 4;
        }
        emit(Op::MakeFunction, code_index(code), flags);
    }
    void compile_function_object(Node* n, std::string_view name) {
        int defaults = 0;
        for (Node* p : n->items) {
            if ((ParamKind) p->op == ParamKind::Positional && p->a) {
                expr(p->a);
                ++defaults;
            }
        }
        int flags = 0;
        if (defaults) {
            emit(Op::BuildTuple, defaults);
            flags |= 1;
        }
        int keyword_only = 0;
        for (Node* p : n->items) {
            if ((ParamKind) p->op != ParamKind::KeywordOnly) continue;
            ++keyword_only;
            if (p->a) {
                expr(p->a);
            } else {
                if (empty_constant < 0) empty_constant = program.constant(Value());
                emit(Op::LoadConst, empty_constant);
            }
        }
        if (keyword_only) {
            emit(Op::BuildTuple, keyword_only);
            flags |= 2;
        }
        Code* code = new_code(child_qualname(name));
        Unit* outer = u;
        Unit unit{code, n->scope};
        u = &unit;
        unit.line = n->line;
        prepare_slots(code, n->scope, &n->items);
        if (n->kind == K::Lambda) {
            expr(n->a);
            emit(Op::Return);
        } else {
            block(n->a);
            load_none();
            emit(Op::Return);
        }
        finish(unit);
        u = outer;
        make_closure_and_function(code, n->scope, flags);
    }
    void compile_comprehension(Node* n) {
        const char* label = n->kind == K::ListComp ? "<listcomp>" : n->kind == K::SetComp ? "<setcomp>" : n->kind == K::DictComp ? "<dictcomp>" : "<genexpr>";
        Code* code = new_code(child_qualname(label));
        Unit* outer = u;
        Unit unit{code, n->scope};
        u = &unit;
        unit.line = n->line;
        code->slots.push_back(".0");
        code->argcount = 1;
        Scope* scope = n->scope;
        prepare_comprehension_slots(code, scope);
        code->generator = n->kind == K::GenExp;
        if (n->kind == K::ListComp) emit(Op::BuildList, 0);
        else if (n->kind == K::SetComp) emit(Op::BuildSet, 0);
        else if (n->kind == K::DictComp) emit(Op::BuildDict, 0);
        comprehension_clause(n, 0);
        if (n->kind == K::GenExp) load_none();
        emit(Op::Return);
        finish(unit);
        u = outer;
        make_closure_and_function(code, scope, 0);
        expr(n->items[0]->b);
        emit(Op::GetIter);
        emit(Op::Call, 1);
    }
    void prepare_comprehension_slots(Code* code, Scope* scope) {
        for (const auto& [name, flags] : scope->symbols) {
            if (name == ".0" || (flags & (kDeclGlobal | kDeclNonlocal)) || !(flags & kBound) || (flags & kFree)) continue;
            code->slots.push_back(name);
        }
        for (std::size_t i = 0; i < code->slots.size(); ++i) {
            const auto hit = scope->symbols.find(code->slots[i]);
            if (hit != scope->symbols.end() && (hit->second & kCell)) code->cells.push_back((int) i);
        }
        code->nlocals = (int) code->slots.size();
        for (const auto& [name, flags] : scope->symbols) {
            if (flags & kFree) {
                code->slots.push_back(name);
                ++code->nfree;
            }
        }
        code->nslots = (int) code->slots.size();
    }
    void comprehension_clause(Node* n, std::size_t i) {
        Node* clause = n->items[i];
        if (i == 0) {
            emit(Op::LoadFast, 0);
        } else {
            expr(clause->b);
            emit(Op::GetIter);
        }
        const int start = new_label(), done = new_label();
        bind(start);
        jump(Op::ForIter, done);
        assign(clause->a);
        for (Node* test : clause->items) {
            expr(test);
            jump(Op::JumpIfFalse, start);
        }
        if (i + 1 < n->items.size()) {
            comprehension_clause(n, i + 1);
        } else {
            const int collect = (int) n->items.size() + 1;
            switch (n->kind) {
                case K::ListComp:
                    expr(n->a);
                    emit(Op::ListAppend, collect);
                    break;
                case K::SetComp:
                    expr(n->a);
                    emit(Op::SetAdd, collect);
                    break;
                case K::DictComp:
                    expr(n->a);
                    expr(n->b);
                    emit(Op::MapAdd, collect);
                    break;
                default:
                    expr(n->a);
                    emit(Op::Yield);
            }
        }
        jump(Op::Jump, start);
        bind(done);
    }

    void expr(Node* n) {
        check_stack();
        if (n->line) u->line = n->line;
        switch (n->kind) {
            case K::Constant: emit(Op::LoadConst, n->index); return;
            case K::Name: load_name(n->name, n->line); return;
            case K::Binary: {
                expr(n->a);
                expr(n->b);
                emit_binary((Bin) n->op, false);
                return;
            }
            case K::Unary:
                expr(n->a);
                emit((P) n->op == P::Minus ? Op::Negate : (P) n->op == P::Plus ? Op::Positive : Op::Invert);
                return;
            case K::Not:
                expr(n->a);
                emit(Op::Not);
                return;
            case K::BoolOp: {
                const int done = new_label();
                for (std::size_t i = 0; i + 1 < n->items.size(); ++i) {
                    expr(n->items[i]);
                    jump(n->op ? Op::JumpIfTrueOrPop : Op::JumpIfFalseOrPop, done);
                }
                expr(n->items.back());
                bind(done);
                return;
            }
            case K::Compare: compare(n); return;
            case K::IfExp: {
                const int otherwise = new_label(), done = new_label();
                expr(n->a);
                jump(Op::JumpIfFalse, otherwise);
                expr(n->b);
                jump(Op::Jump, done);
                --u->depth;
                bind(otherwise);
                expr(n->c);
                bind(done);
                return;
            }
            case K::Lambda: compile_function_object(n, "<lambda>"); return;
            case K::Call: call(n); return;
            case K::Attribute:
                expr(n->a);
                emit(Op::LoadAttr, attr_index(n->name));
                return;
            case K::Subscript:
                expr(n->a);
                expr(n->b);
                emit(Op::Subscr);
                return;
            case K::Slice:
                if (n->a) expr(n->a);
                else load_none();
                if (n->b) expr(n->b);
                else load_none();
                if (n->c) expr(n->c);
                else load_none();
                emit(Op::BuildSlice);
                return;
            case K::Tuple: display(n, Op::BuildTuple); return;
            case K::List: display(n, Op::BuildList); return;
            case K::Set: display(n, Op::BuildSet); return;
            case K::Dict: dict_display(n); return;
            case K::ListComp: case K::SetComp: case K::GenExp: case K::DictComp: compile_comprehension(n); return;
            case K::Starred: syntax_error(n->line, "can't use starred expression here");
            case K::Walrus:
                expr(n->b);
                emit(Op::Dup);
                store_name(n->a->name);
                return;
            case K::FString: {
                for (Node* part : n->items) {
                    if (part->kind == K::Constant) emit(Op::LoadConst, part->index);
                    else {
                        expr(part->a);
                        if (part->b) expr(part->b);
                        emit(Op::FormatValue, part->op, part->b ? 1 : 0);
                    }
                }
                emit(Op::BuildString, (int) n->items.size());
                return;
            }
            default: syntax_error(n->line, "invalid syntax");
        }
    }
    void emit_binary(Bin op, bool inplace) {
        const int flag = inplace ? 1 : 0;
        if (op == Bin::Add) emit(Op::Add, 0, flag);
        else if (op == Bin::Sub) emit(Op::Sub, 0, flag);
        else if (op == Bin::Mul) emit(Op::Mul, 0, flag);
        else emit(Op::Binary, (int) op, flag);
    }
    void compare(Node* n) {
        expr(n->a);
        if (n->items.size() == 1) {
            expr(n->items[0]);
            emit(Op::Compare, n->ops[0]);
            return;
        }
        const int cleanup = new_label(), done = new_label();
        for (std::size_t i = 0; i + 1 < n->items.size(); ++i) {
            expr(n->items[i]);
            emit(Op::Dup);
            emit(Op::Rot3);
            emit(Op::Compare, n->ops[i]);
            jump(Op::JumpIfFalseOrPop, cleanup);
        }
        expr(n->items.back());
        emit(Op::Compare, n->ops.back());
        const int after = u->depth;
        jump(Op::Jump, done);
        ++u->depth;
        bind(cleanup);
        u->depth = after + 1;
        emit(Op::Rot2);
        emit(Op::Pop);
        bind(done);
        u->depth = after;
    }
    void display(Node* n, Op build) {
        bool starred = false;
        for (const Node* item : n->items) starred = starred || item->kind == K::Starred;
        // A long literal is built in place: its frame must not need a stack slot per element.
        if (!starred && n->items.size() <= 256) {
            for (Node* item : n->items) expr(item);
            emit(build, (int) n->items.size());
            return;
        }
        emit(build == Op::BuildSet ? Op::BuildSet : Op::BuildList, 0);
        for (Node* item : n->items) {
            if (item->kind == K::Starred) {
                expr(item->a);
                emit(build == Op::BuildSet ? Op::SetUpdate : Op::ListExtend, 1);
            } else {
                expr(item);
                emit(build == Op::BuildSet ? Op::SetAdd : Op::ListAppend, 1);
            }
        }
        if (build == Op::BuildTuple) emit(Op::ListToTuple);
    }
    void dict_display(Node* n) {
        emit(Op::BuildDict, 0);
        for (std::size_t i = 0; i < n->items.size(); ++i) {
            if (n->items[i]) {
                expr(n->items[i]);
                expr(n->more[i]);
                emit(Op::MapAdd, 1);
            } else {
                expr(n->more[i]);
                emit(Op::DictUpdate, 1);
            }
        }
    }
    void call(Node* n) {
        bool starred = false, double_star = false;
        for (const Node* item : n->items) starred = starred || item->kind == K::Starred;
        for (const Node* keyword : n->more) double_star = double_star || keyword->name.empty();
        if (starred || double_star) {
            expr(n->a);
            emit(Op::BuildList, 0);
            for (Node* item : n->items) {
                if (item->kind == K::Starred) {
                    expr(item->a);
                    emit(Op::ListExtend, 1, 1);
                } else {
                    expr(item);
                    emit(Op::ListAppend, 1);
                }
            }
            if (!n->more.empty()) {
                emit(Op::BuildDict, 0);
                for (Node* keyword : n->more) {
                    if (keyword->name.empty()) {
                        expr(keyword->a);
                        emit(Op::DictMerge, 1);
                    } else {
                        emit(Op::LoadConst, program.constant(text(keyword->name)));
                        expr(keyword->a);
                        emit(Op::MapAdd, 1);
                    }
                }
            }
            u->line = n->line;
            emit(Op::CallEx, 0, n->more.empty() ? 0 : 1);
            return;
        }
        const bool method = n->a->kind == K::Attribute;
        if (method) expr(n->a->a);
        else expr(n->a);
        for (Node* item : n->items) expr(item);
        for (Node* keyword : n->more) expr(keyword->a);
        if (!n->more.empty()) {
            Vals names;
            for (const Node* keyword : n->more) names.push_back(text(keyword->name));
            emit(Op::KwNames, program.constant(make_tuple(std::move(names))));
        }
        const int total = (int) (n->items.size() + n->more.size());
        u->line = n->line;
        if (method) emit(Op::CallMethod, attr_index(n->a->name), total);
        else emit(Op::Call, total);
    }
    void assign(Node* t) {
        switch (t->kind) {
            case K::Name: store_name(t->name); return;
            case K::Subscript:
                expr(t->a);
                expr(t->b);
                emit(Op::StoreSubscr);
                return;
            case K::Attribute:
                expr(t->a);
                emit(Op::StoreAttr, attr_index(t->name));
                return;
            case K::Tuple: case K::List: {
                int star = -1;
                for (std::size_t i = 0; i < t->items.size(); ++i)
                    if (t->items[i]->kind == K::Starred) star = (int) i;
                if (star < 0) emit(Op::Unpack, (int) t->items.size());
                else emit(Op::UnpackEx, star, (int) t->items.size() - star - 1);
                for (Node* item : t->items) assign(item->kind == K::Starred ? item->a : item);
                return;
            }
            default: syntax_error(t->line, "cannot assign to expression");
        }
    }
    void delete_target(Node* t) {
        if (t->kind == K::Name) {
            delete_name(t->name);
        } else if (t->kind == K::Subscript) {
            expr(t->a);
            expr(t->b);
            emit(Op::DeleteSubscr);
        } else if (t->kind == K::Attribute) {
            expr(t->a);
            emit(Op::DeleteAttr, attr_index(t->name));
        } else {
            for (Node* item : t->items) delete_target(item);
        }
    }
    void block(Node* n) {
        for (Node* s : n->items) statement(s);
    }
    // Leaves every context above `keep`, running what each one owes on the way out.
    void exit_contexts(std::size_t keep) {
        for (std::size_t i = u->ctx.size(); i-- > keep;) {
            const Ctx ctx = u->ctx[i];
            switch (ctx.kind) {
                case CtxKind::Loop:
                    if (ctx.for_loop) emit(Op::Pop);
                    break;
                case CtxKind::Try: emit(Op::PopBlock); break;
                case CtxKind::Handler: emit(Op::PopExcept); break;
                case CtxKind::FinallyHandler:
                    emit(Op::Pop);
                    emit(Op::PopExcept);
                    break;
                case CtxKind::AsName:
                    emit(Op::PopBlock);
                    load_none();
                    store_name(ctx.name);
                    delete_name(ctx.name);
                    break;
                case CtxKind::Finally: {
                    emit(Op::PopBlock);
                    std::vector<Ctx, Alloc<Ctx>> rest(u->ctx.begin() + (std::ptrdiff_t) i, u->ctx.end());
                    u->ctx.resize(i, Ctx{CtxKind::Try});
                    block(ctx.body);
                    u->ctx.insert(u->ctx.end(), rest.begin(), rest.end());
                    break;
                }
            }
        }
    }
    void statement(Node* n) {
        check_stack();
        u->line = n->line;
        switch (n->kind) {
            case K::ExprStmt:
                expr(n->a);
                emit(Op::Pop);
                return;
            case K::Assign:
                expr(n->a);
                for (std::size_t i = 0; i < n->items.size(); ++i) {
                    if (i + 1 < n->items.size()) emit(Op::Dup);
                    assign(n->items[i]);
                }
                return;
            case K::AugAssign:
                if (n->a->kind == K::Name) {
                    load_name(n->a->name, n->line);
                    expr(n->b);
                    emit_binary((Bin) n->op, true);
                    store_name(n->a->name);
                } else if (n->a->kind == K::Attribute) {
                    expr(n->a->a);
                    emit(Op::Dup);
                    emit(Op::LoadAttr, attr_index(n->a->name));
                    expr(n->b);
                    emit_binary((Bin) n->op, true);
                    emit(Op::Rot2);
                    emit(Op::StoreAttr, attr_index(n->a->name));
                } else {
                    expr(n->a->a);
                    expr(n->a->b);
                    emit(Op::Dup2);
                    emit(Op::Subscr);
                    expr(n->b);
                    emit_binary((Bin) n->op, true);
                    emit(Op::Rot3);
                    emit(Op::StoreSubscr);
                }
                return;
            case K::AnnAssign:
                if (n->b) {
                    expr(n->b);
                    assign(n->a);
                }
                return;
            case K::If: {
                const int done = new_label();
                for (std::size_t i = 0; i + 1 < n->items.size(); i += 2) {
                    const int next = new_label();
                    expr(n->items[i]);
                    jump(Op::JumpIfFalse, next);
                    block(n->items[i + 1]);
                    jump(Op::Jump, done);
                    bind(next);
                }
                if (n->a) block(n->a);
                bind(done);
                return;
            }
            case K::While: {
                const int start = new_label(), otherwise = new_label(), done = new_label();
                bind(start);
                expr(n->a);
                jump(Op::JumpIfFalse, otherwise);
                Ctx ctx{CtxKind::Loop};
                ctx.break_label = done;
                ctx.continue_label = start;
                u->ctx.push_back(ctx);
                block(n->b);
                u->ctx.pop_back();
                jump(Op::Jump, start);
                bind(otherwise);
                if (n->c) block(n->c);
                bind(done);
                return;
            }
            case K::For: {
                const int start = new_label(), otherwise = new_label(), done = new_label();
                expr(n->b);
                emit(Op::GetIter);
                const int after = u->depth - 1;
                bind(start);
                u->line = n->line;
                jump(Op::ForIter, otherwise);
                assign(n->a);
                Ctx ctx{CtxKind::Loop};
                ctx.for_loop = true;
                ctx.break_label = done;
                ctx.continue_label = start;
                u->ctx.push_back(ctx);
                block(n->c);
                u->ctx.pop_back();
                jump(Op::Jump, start);
                bind(otherwise);
                if (n->d) block(n->d);
                u->depth = after;
                u->label_depth[(std::size_t) done] = after;
                bind(done);
                return;
            }
            case K::Break: case K::Continue: {
                std::size_t i = u->ctx.size();
                while (i > 0 && u->ctx[i - 1].kind != CtxKind::Loop) --i;
                if (i == 0) syntax_error(n->line, n->kind == K::Break ? "'break' outside loop" : "'continue' not properly in loop");
                const Ctx loop = u->ctx[i - 1];
                const int depth = u->depth;
                if (n->kind == K::Break) {
                    exit_contexts(i - 1);
                    jump(Op::Jump, loop.break_label);
                } else {
                    exit_contexts(i);
                    const int saved = u->label_depth[(std::size_t) loop.continue_label];
                    jump(Op::Jump, loop.continue_label);
                    u->label_depth[(std::size_t) loop.continue_label] = saved;
                }
                u->depth = depth;
                return;
            }
            case K::Pass: return;
            case K::Def: {
                if (n->c) expr(n->c);
                compile_function_object(n, n->name);
                if (n->c) emit(Op::Call, 1);
                store_name(n->name);
                return;
            }
            case K::Class: {
                emit(Op::BuildClass, attr_index(n->name));
                for (Node* item : n->a->items) {
                    if (item->kind != K::Def) continue;
                    if (item->c) expr(item->c);
                    compile_function_object(item, std::string(n->name) + "." + std::string(item->name));
                    if (item->c) emit(Op::Call, 1);
                    emit(Op::AddMethod, attr_index(item->name));
                }
                store_name(n->name);
                return;
            }
            case K::Return: {
                if (n->a) expr(n->a);
                else load_none();
                bool unwinding = false;
                for (const Ctx& ctx : u->ctx) unwinding = unwinding || ctx.kind == CtxKind::Finally || ctx.kind == CtxKind::AsName;
                if (!unwinding) {
                    emit(Op::Return);
                    ++u->depth;
                    return;
                }
                if (u->ret_slot < 0) {
                    u->ret_slot = u->code->nslots++;
                    u->code->slots.push_back(".return");
                }
                const int depth = u->depth;
                emit(Op::StoreFast, u->ret_slot);
                exit_contexts(0);
                emit(Op::LoadFast, u->ret_slot);
                emit(Op::Return);
                u->depth = depth - 1;
                return;
            }
            case K::Raise:
                if (n->a) expr(n->a);
                if (n->b) expr(n->b);
                u->line = n->line;
                emit(Op::Raise, (n->a ? 1 : 0) + (n->b ? 1 : 0));
                return;
            case K::Global: case K::Nonlocal: return;
            case K::Assert: {
                const int done = new_label();
                expr(n->a);
                jump(Op::JumpIfTrue, done);
                const int depth = u->depth;
                emit(Op::LoadConst, program.constant(Value::type(T_AssertionError)));
                if (n->b) {
                    expr(n->b);
                    emit(Op::Call, 1);
                }
                emit(Op::Raise, 1);
                u->depth = depth;
                bind(done);
                return;
            }
            case K::Del:
                for (Node* t : n->items) delete_target(t);
                return;
            case K::Import:
                for (Node* alias : n->items) {
                    emit(Op::Import, attr_index(alias->name), alias->names.empty() ? 1 : 0);
                    const std::string_view whole = alias->name;
                    store_name(alias->names.empty() ? whole.substr(0, whole.find('.')) : alias->names[0]);
                }
                return;
            case K::ImportFrom:
                emit(Op::Import, attr_index(n->name));
                for (Node* alias : n->items) {
                    emit(Op::ImportFrom, attr_index(alias->name));
                    store_name(alias->names.empty() ? alias->name : alias->names[0]);
                }
                emit(Op::Pop);
                return;
            case K::Try: compile_try(n); return;
            default: syntax_error(n->line, "invalid syntax");
        }
    }
    void compile_try(Node* n) {
        if (!n->c) {
            try_except(n);
            return;
        }
        const int handler = new_label(), done = new_label();
        const int base = u->depth;
        u->label_depth[(std::size_t) handler] = base + 1;
        jump_setup(handler);
        Ctx ctx{CtxKind::Finally};
        ctx.body = n->c;
        u->ctx.push_back(ctx);
        if (n->items.empty()) block(n->a);
        else try_except(n);
        u->ctx.pop_back();
        emit(Op::PopBlock);
        block(n->c);
        jump(Op::Jump, done);
        u->depth = base + 1;
        bind(handler);
        emit(Op::PushHandled);
        u->ctx.push_back(Ctx{CtxKind::FinallyHandler});
        block(n->c);
        u->ctx.pop_back();
        emit(Op::Reraise);
        u->depth = base;
        bind(done);
    }
    void jump_setup(int label) {
        const int at = emit(Op::SetupTry);
        u->patches.emplace_back(at, label);
    }
    void try_except(Node* n) {
        const int handler = new_label(), done = new_label();
        const int base = u->depth;
        u->label_depth[(std::size_t) handler] = base + 1;
        jump_setup(handler);
        u->ctx.push_back(Ctx{CtxKind::Try});
        block(n->a);
        u->ctx.pop_back();
        emit(Op::PopBlock);
        if (n->b) block(n->b);
        jump(Op::Jump, done);
        u->depth = base + 1;
        bind(handler);
        emit(Op::PushHandled);
        for (std::size_t i = 0; i < n->items.size(); ++i) {
            Node* h = n->items[i];
            u->line = h->line;
            if (!h->a && i + 1 != n->items.size()) syntax_error(h->line, "default 'except:' must be last");
            const int next = new_label();
            if (h->a) {
                emit(Op::Dup);
                expr(h->a);
                emit(Op::ExcMatch);
                jump(Op::JumpIfFalse, next);
            }
            if (!h->name.empty()) store_name(h->name);
            else emit(Op::Pop);
            u->ctx.push_back(Ctx{CtxKind::Handler});
            if (!h->name.empty()) {
                const int cleanup = new_label();
                u->label_depth[(std::size_t) cleanup] = u->depth + 1;
                jump_setup(cleanup);
                Ctx as{CtxKind::AsName};
                as.name = h->name;
                u->ctx.push_back(as);
                block(h->b);
                u->ctx.pop_back();
                emit(Op::PopBlock);
                load_none();
                store_name(h->name);
                delete_name(h->name);
                u->ctx.pop_back();
                emit(Op::PopExcept);
                jump(Op::Jump, done);
                u->depth = base + 1;
                bind(cleanup);
                load_none();
                store_name(h->name);
                delete_name(h->name);
                emit(Op::Reraise);
            } else {
                block(h->b);
                u->ctx.pop_back();
                emit(Op::PopExcept);
                jump(Op::Jump, done);
            }
            u->depth = base + 1;
            bind(next);
        }
        emit(Op::Reraise);
        u->depth = base;
        bind(done);
    }
    Code* compile_module(Node* tree) {
        ScopeBuilder builder{scopes};
        module_scope = builder.open(Scope::Kind::Module, nullptr);
        builder.module = module_scope;
        builder.walk(module_scope, tree);
        builder.resolve(module_scope, 1);
        Code* code = new_code("<module>");
        Unit unit{code, module_scope};
        u = &unit;
        block(tree);
        load_none();
        emit(Op::Return);
        finish(unit);
        u = nullptr;
        return code;
    }
};

struct GenObj;
struct Frame {
    const Code* code = nullptr;
    const Instr* ip = nullptr;
    Value* base = nullptr;
    Value* sp = nullptr;
    Value func;
    GenObj* gen = nullptr;
    int block = -1;           // offset of the top Block from `base`
    uint32_t handled = 0;
};
struct GenObj : IterObj {
    Vals storage;
    Frame saved;
    bool running = false, finished = false;
    GenObj() : IterObj(T_generator) {}
    void clear() override {
        storage.clear();
        saved.func.reset();
        finished = true;
    }
    bool next(Value& out) override;
};
struct VM;
inline thread_local VM* vm = nullptr;
inline Value import_module(std::string_view name);
inline Value construct(int type, Value* a, int n, const TupleObj* kw);
inline Value builtin_value(std::string_view name);
inline Value make_cached_call(CachedObj* f, Value* args, int n, const TupleObj* kw);
inline Value call_native(const NativeDef* d, Value* a, int n, const TupleObj* kw, int self_args = 0) {
    Call c{a, n, kw};
    if (n < d->min || (d->max >= 0 && n > d->max)) {
        const int given = n - self_args, low = d->min - self_args, high = d->max - self_args;
        const std::string who = self_args ? std::string(type_name(a[0])) + "." + d->name : std::string(d->name);
        if (low == high && low == 1) raise(T_TypeError, who + "() takes exactly one argument (" + std::to_string(given) + " given)");
        if (low == high && low == 0) raise(T_TypeError, who + "() takes no arguments (" + std::to_string(given) + " given)");
        if (d->keywords && n > d->max) raise(T_TypeError, std::string(d->name) + "() takes at most " + std::to_string(high) + " argument" + (high == 1 ? "" : "s") + " (" + std::to_string(given) + " given)");
        const int bound = n < d->min ? low : high;
        raise(T_TypeError, std::string(d->name) + " expected " + (low == high ? "" : n < d->min ? "at least " : "at most ") + std::to_string(bound) + " argument" + (bound == 1 ? "" : "s") + ", got " + std::to_string(given));
    }
    if (c.nkw() && !d->keywords) raise(T_TypeError, std::string(d->name) + "() takes no keyword arguments");
    return d->fn(c);
}

struct VM {
    Program& program;
    Compiler& compiler;
    Vals stack, globals, builtins, handled;
    Value* top;
    Value* stack_end;
    std::vector<Frame, Alloc<Frame>> frames;
    std::size_t depth = 0;
    const TupleObj* pending_kw = nullptr;
    VM(Program& p, Compiler& c) : program(p), compiler(c) {
        stack.resize(kValueStack);
        top = stack.data();
        stack_end = top + stack.size();
        frames.resize((std::size_t) rt->limits.recursion_depth + 2);
        globals.resize(compiler.global_names.size());
        builtins.resize(compiler.global_names.size());
        for (std::size_t i = 0; i < compiler.global_names.size(); ++i) builtins[i] = builtin_value(compiler.global_names[i]);
    }
    [[noreturn]] void too_deep() { throw Stop{Status::RecursionLimit, "RecursionError: depth limit " + std::to_string(rt->limits.recursion_depth)}; }
    int line_of(const Frame& f) const {
        const std::ptrdiff_t at = f.ip - f.code->ins.data() - 1;
        return at >= 0 && at < (std::ptrdiff_t) f.code->lines.size() ? f.code->lines[(std::size_t) at] : 0;
    }
    int current_line() const { return depth ? line_of(frames[depth - 1]) : 0; }

    [[noreturn]] void bad_positional(const FunctionObj* fn, int given, int keyword_only) {
        const Code* c = fn->code;
        const int required = c->argcount - (int) fn->defaults.size();
        std::string takes = fn->defaults.empty() ? std::to_string(c->argcount) : "from " + std::to_string(required) + " to " + std::to_string(c->argcount);
        const bool plural = !fn->defaults.empty() || c->argcount != 1;
        std::string extra;
        if (keyword_only) extra = std::string(" positional argument") + (given != 1 ? "s" : "") + " (and " + std::to_string(keyword_only) + " keyword-only argument" + (keyword_only != 1 ? "s" : "") + ")";
        raise(T_TypeError, c->qualname + "() takes " + takes + " positional argument" + (plural ? "s" : "") + " but " + std::to_string(given) + extra + (given == 1 && !keyword_only ? " was" : " were") + " given");
    }
    [[noreturn]] void missing(const Code* c, const std::vector<std::string_view, Alloc<std::string_view>>& names, const char* kind) {
        std::string list;
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (i) list += names.size() == 2 ? " and " : i + 1 == names.size() ? ", and " : ", ";
            list += "'" + std::string(names[i]) + "'";
        }
        raise(T_TypeError, c->qualname + "() missing " + std::to_string(names.size()) + " required " + kind + " argument" + (names.size() == 1 ? "" : "s") + ": " + list);
    }
    // CPython's argument binding. Takes the arguments out of `args`.
    void bind_args(const FunctionObj* fn, Value* locals, Value* args, int npos, const TupleObj* kw) {
        const Code* c = fn->code;
        const int total = c->argcount, named = total + c->kwonly;
        const int copied = std::min(npos, total);
        for (int i = 0; i < copied; ++i) locals[i] = std::move(args[i]);
        if (c->star >= 0) {
            Vals rest;
            if (npos > total) {
                rest.reserve((std::size_t) (npos - total));
                for (int i = total; i < npos; ++i) rest.push_back(std::move(args[i]));
            }
            locals[c->star] = make_tuple(std::move(rest));
        }
        if (c->double_star >= 0) locals[c->double_star] = make_dict();
        const int nkw = kw ? (int) kw->v.size() : 0;
        for (int k = 0; k < nkw; ++k) {
            const Value& name = kw->v[(std::size_t) k];
            int at = -1;
            for (int i = 0; i < named; ++i) {
                if (equals(name, c->slots[(std::size_t) i])) {
                    at = i;
                    break;
                }
            }
            if (at >= 0) {
                if (!locals[at].empty()) raise(T_TypeError, c->qualname + "() got multiple values for argument '" + utf8(str_of(name)) + "'");
                locals[at] = std::move(args[npos + k]);
            } else if (c->double_star >= 0) {
                locals[c->double_star].as<DictObj>()->t.set(name, std::move(args[npos + k]));
            } else {
                raise(T_TypeError, c->qualname + "() got an unexpected keyword argument '" + utf8(str_of(name)) + "'");
            }
        }
        if (npos > total && c->star < 0) {
            int keyword_only_given = 0;
            for (int i = 0; i < c->kwonly; ++i) keyword_only_given += locals[total + i].empty() ? 0 : 1;
            bad_positional(fn, npos, keyword_only_given);
        }
        const int first_default = total - (int) fn->defaults.size();
        std::vector<std::string_view, Alloc<std::string_view>> absent;
        for (int i = npos < total ? npos : total; i < total; ++i) {
            if (!locals[i].empty()) continue;
            if (i >= first_default) locals[i] = fn->defaults[(std::size_t) (i - first_default)];
            else absent.push_back(c->slots[(std::size_t) i]);
        }
        if (!absent.empty()) missing(c, absent, "positional");
        for (int i = 0; i < c->kwonly; ++i) {
            Value& slot = locals[total + i];
            if (!slot.empty()) continue;
            if (!fn->kw_defaults[(std::size_t) i].empty()) slot = fn->kw_defaults[(std::size_t) i];
            else absent.push_back(c->slots[(std::size_t) (total + i)]);
        }
        if (!absent.empty()) missing(c, absent, "keyword-only");
    }
    void setup_locals(const FunctionObj* fn, Value* locals) {
        const Code* c = fn->code;
        for (const int slot : c->cells) {
            auto* cell = new CellObj;
            Value holder(cell);
            cell->v = std::move(locals[slot]);
            locals[slot] = std::move(holder);
        }
        for (int i = 0; i < c->nfree; ++i) locals[c->nlocals + i] = fn->closure[(std::size_t) i];
    }
    // Pushes a frame, or returns the generator that will own it.
    Value enter(const Value& func, Value* args, int npos, const TupleObj* kw) {
        const auto* fn = func.as<FunctionObj>();
        const Code* c = fn->code;
        if (depth > (std::size_t) rt->limits.recursion_depth) too_deep();
        const std::size_t size = (std::size_t) c->frame_size();
        if (c->generator) {
            auto* gen = new GenObj;
            Value holder(gen);
            gen->storage.resize(size);
            Value* locals = gen->storage.data();
            bind_args(fn, locals, args, npos, kw);
            setup_locals(fn, locals);
            gen->saved.code = c;
            gen->saved.ip = c->ins.data();
            gen->saved.base = locals;
            gen->saved.sp = locals + c->nslots;
            gen->saved.func = func;
            return holder;
        }
        if (size > (std::size_t) (stack_end - top)) throw Stop{Status::RecursionLimit, "RecursionError: the frames fill the value stack"};
        Value* locals = top;
        struct Unbind {
            Value* locals;
            std::size_t n;
            ~Unbind() {
                for (std::size_t i = 0; i < n; ++i) locals[i].reset();
            }
        } unbind{locals, (std::size_t) c->nslots};
        bind_args(fn, locals, args, npos, kw);
        setup_locals(fn, locals);
        unbind.n = 0;
        top += size;
        Frame& f = frames[depth++];
        f.code = c;
        f.ip = c->ins.data();
        f.base = locals;
        f.sp = locals + c->nslots;
        f.func = func;
        f.gen = nullptr;
        f.block = -1;
        f.handled = (uint32_t) handled.size();
        return Value();
    }
    void leave(Frame* f) {
        for (Value* p = f->base; p < f->sp; ++p) p->reset();
        if (handled.size() > f->handled) handled.resize(f->handled);
        if (f->gen) f->gen->finished = true;
        else top = f->base;
        f->func.reset();
        --depth;
    }
    bool unwind(Value& exc, std::size_t stop) {
        while (depth > stop) {
            Frame* f = &frames[depth - 1];
            if (f->block >= 0) {
                Value* block = f->base + f->block;
                for (Value* p = block + 1; p < f->sp; ++p) p->reset();
                const int handler = (int) (block->i >> 32);
                const int previous = (int) (uint32_t) block->i - 1;
                if (handled.size() > block->aux) handled.resize(block->aux);
                f->block = previous;
                *block = std::move(exc);
                f->sp = block + 1;
                f->ip = f->code->ins.data() + handler;
                return true;
            }
            leave(f);
        }
        return false;
    }
    // A catch block runs on top of the throw point, so rethrowing from inside one would pile a dispatch per level
    // onto the stack: the exception leaves the catch first and is thrown again from here.
    Value run(std::size_t stop) {
        for (;;) {
            Value passing;
            try {
                return execute(stop);
            } catch (PyError& e) {
                auto* obj = e.exc.as<ExcObj>();
                if (!obj->line) obj->line = current_line();
                rt->steps += 64;   // a C++ throw costs about 1.5 us, not one step
                if (unwind(e.exc, stop)) continue;
                passing = std::move(e.exc);
            }
            throw PyError{std::move(passing)};
        }
    }
    static Vals with_self(const Value& self, Value* args, int n, const TupleObj* kw) {
        const int nkw = kw ? (int) kw->v.size() : 0;
        Vals full((std::size_t) (n + nkw + 1));
        full[0] = self;
        for (int i = 0; i < n + nkw; ++i) full[(std::size_t) i + 1] = std::move(args[i]);
        return full;
    }
    Value instantiate(const Value& cls, Value* args, int n, const TupleObj* kw) {
        auto* c = cls.as<ClassObj>();
        Value self(new InstanceObj(cls));
        const Value* init = find_attr(c->methods, "__init__");
        if (!init) {
            if (n || kw) raise(T_TypeError, std::string(c->name) + "() takes no arguments");
            return self;
        }
        const Value fn = *init;
        Vals full = with_self(self, args, n, kw);
        const Value out = call(fn, full.data(), n + 1, kw);
        if (!out.is_none()) raise(T_TypeError, "__init__() should return None, not " + quoted(out));
        return self;
    }
    Value call(const Value& f, Value* args, int n, const TupleObj* kw) {
        switch (f.tag) {
            case Tag::Function: {
                Nest nest;
                const std::size_t at = depth;
                Value gen = enter(f, args, n, kw);
                if (!gen.empty()) return gen;
                return run(at);
            }
            case Tag::Native: return call_native(f.native, args, n, kw);
            case Tag::Type: return construct((int) f.i, args, n, kw);
            case Tag::Class: return instantiate(f, args, n, kw);
            case Tag::Method: {
                auto* m = f.as<MethodObj>();
                const Value fn = m->fn;
                Vals full = with_self(m->self, args, n, kw);
                return call(fn, full.data(), n + 1, kw);
            }
            case Tag::Bound: {
                auto* b = f.as<BoundObj>();
                const int nkw = kw ? (int) kw->v.size() : 0;
                Vals full((std::size_t) (n + nkw + 1));
                full[0] = b->self;
                for (int i = 0; i < n + nkw; ++i) full[(std::size_t) i + 1] = std::move(args[i]);
                return call_native(b->method, full.data(), n + 1, kw, 1);
            }
            case Tag::Cached: return make_cached_call(f.as<CachedObj>(), args, n, kw);
            case Tag::Stub: refuse("calling a typing object");
            default: raise(T_TypeError, quoted(f) + " object is not callable");
        }
    }

    void clear_range(Value* from, Value* to) {
        for (Value* p = from; p < to; ++p) p->reset();
    }
    void binary_top(Frame* f, Bin op, bool inplace) {
        Value* sp = f->sp;
        Value out = binary(op, sp[-2], sp[-1], inplace);
        sp[-1].reset();
        sp[-2] = std::move(out);
        f->sp = sp - 1;
    }
    static bool identical(const Value& a, const Value& b) {
        if (a.tag != b.tag) return false;
        switch (a.tag) {
            case Tag::None: case Tag::Ellipsis: case Tag::Stub: case Tag::Empty: return true;
            case Tag::Bool: return a.b == b.b;
            case Tag::Type: case Tag::Module: return a.i == b.i;
            case Tag::Native: return a.native == b.native;
            case Tag::Int: case Tag::Float: refuse("`is` on numbers, which CPython answers by its object cache");
            case Tag::Big: case Tag::Str: case Tag::Tuple: case Tag::Fraction:
                if (a.o == b.o) return true;
                refuse(std::string("`is` on ") + type_name(a) + " values, which CPython answers by interning");
            default: return a.o == b.o;
        }
    }
    bool compare(CmpOp op, const Value& a, const Value& b) {
        switch (op) {
            case CmpOp::Lt: return order(Cmp::Lt, a, b);
            case CmpOp::Le: return order(Cmp::Le, a, b);
            case CmpOp::Gt: return order(Cmp::Gt, a, b);
            case CmpOp::Ge: return order(Cmp::Ge, a, b);
            case CmpOp::Eq: return eq(a, b);
            case CmpOp::Ne: return !eq(a, b);
            case CmpOp::In: return contains(b, a);
            case CmpOp::NotIn: return !contains(b, a);
            case CmpOp::Is: return identical(a, b);
            default: return !identical(a, b);
        }
    }
    void compare_top(Frame* f, CmpOp op) {
        Value* sp = f->sp;
        const bool out = compare(op, sp[-2], sp[-1]);
        sp[-1].reset();
        sp[-2] = Value::boolean(out);
        f->sp = sp - 1;
    }
    void unary_top(Frame* f, Op op) {
        Value& v = f->sp[-1];
        if (op == Op::Negate) {
            if (is_int(v)) v = int_negate(v);
            else if (v.tag == Tag::Float) v = Value::real(-v.d);
            else if (v.tag == Tag::Fraction) v = frac_make(int_negate(v.as<FractionObj>()->num), v.as<FractionObj>()->den);
            else if (v.tag == Tag::Dict && v.as<DictObj>()->kind == DictKind::Counter) v = counter_binary(Bin::Sub, make_dict(DictKind::Counter), v);
            else raise(T_TypeError, "bad operand type for unary -: " + quoted(v));
        } else if (op == Op::Positive) {
            if (v.tag == Tag::Bool) v = Value::integer(small_of(v));
            else if (!is_number(v) && v.tag != Tag::Fraction) {
                if (v.tag == Tag::Dict && v.as<DictObj>()->kind == DictKind::Counter) v = counter_binary(Bin::Add, make_dict(DictKind::Counter), v);
                else raise(T_TypeError, "bad operand type for unary +: " + quoted(v));
            }
        } else {
            if (!is_int(v)) raise(T_TypeError, "bad operand type for unary ~: " + quoted(v));
            v = int_sub(Value::integer(-1), v);
        }
    }
    void build_seq(Frame* f, int n, bool tuple) {
        Value* first = f->sp - n;
        Vals v;
        v.reserve((std::size_t) n);
        for (int i = 0; i < n; ++i) v.push_back(std::move(first[i]));
        Value out = tuple ? make_tuple(std::move(v)) : make_list(std::move(v));
        *first = std::move(out);
        f->sp = first + 1;
    }
    void build_set(Frame* f, int n) {
        Value* first = f->sp - n;
        Value out = make_set();
        for (int i = 0; i < n; ++i) out.as<SetObj>()->t.set(first[i], Value::none());
        clear_range(first, f->sp);
        *first = std::move(out);
        f->sp = first + 1;
    }
    void build_string(Frame* f, int n) {
        Value* first = f->sp - n;
        U32 out;
        for (int i = 0; i < n; ++i) out += str_of(first[i]);
        rt->tick(out.size() >> 6);
        clear_range(first, f->sp);
        *first = text(std::move(out));
        f->sp = first + 1;
    }
    void format_top(Frame* f, int conversion, bool has_spec) {
        Value* sp = f->sp;
        Value* value = has_spec ? sp - 2 : sp - 1;
        Value converted = *value;
        if (conversion == 'r') {
            U32 s;
            repr_to(s, converted);
            converted = text(std::move(s));
        } else if (conversion == 's') {
            converted = str_value(converted);
        } else if (conversion == 'a') {
            U32 s;
            repr_to(s, converted);
            need_ascii(s, "ascii()");
            converted = text(std::move(s));
        }
        Value out = has_spec ? format_value(converted, str_of(sp[-1])) : str_value(converted);
        if (has_spec) sp[-1].reset();
        *value = std::move(out);
        f->sp = value + 1;
    }
    void unpack(Frame* f, int want) {
        Value* slot = f->sp - 1;
        const Value source = std::move(*slot);
        Value* const start = slot;
        if (source.tag == Tag::List || source.tag == Tag::Tuple) {
            const Vals& v = seq_of(source);
            if ((int) v.size() == want) {
                for (int i = 0; i < want; ++i) start[i] = v[(std::size_t) (want - 1 - i)];
                f->sp = start + want;
                return;
            }
            f->sp = start;
            if ((int) v.size() < want) raise(T_ValueError, "not enough values to unpack (expected " + std::to_string(want) + ", got " + std::to_string(v.size()) + ")");
            raise(T_ValueError, "too many values to unpack (expected " + std::to_string(want) + ")");
        }
        f->sp = start;
        if (!is_iterable(source)) raise(T_TypeError, std::string("cannot unpack non-iterable ") + type_name(source) + " object");
        Value it = get_iter(source);
        Vals got;
        Value x;
        while ((int) got.size() <= want && iter_next(it, x)) got.push_back(std::move(x));
        if ((int) got.size() < want) raise(T_ValueError, "not enough values to unpack (expected " + std::to_string(want) + ", got " + std::to_string(got.size()) + ")");
        if ((int) got.size() > want) raise(T_ValueError, "too many values to unpack (expected " + std::to_string(want) + ")");
        for (int i = 0; i < want; ++i) start[i] = std::move(got[(std::size_t) (want - 1 - i)]);
        f->sp = start + want;
    }
    void unpack_star(Frame* f, int before, int after) {
        Value* start = f->sp - 1;
        const Value source = std::move(*start);
        f->sp = start;
        if (!is_iterable(source)) raise(T_TypeError, std::string("cannot unpack non-iterable ") + type_name(source) + " object");
        Value it = get_iter(source);
        Vals all;
        Value x;
        while (iter_next(it, x)) {
            rt->tick();
            all.push_back(std::move(x));
        }
        const int size = (int) all.size();
        if (size < before + after) raise(T_ValueError, "not enough values to unpack (expected at least " + std::to_string(before + after) + ", got " + std::to_string(size) + ")");
        Value* out = start;
        for (int i = 0; i < after; ++i) *out++ = std::move(all[(std::size_t) (size - 1 - i)]);
        Vals middle;
        middle.reserve((std::size_t) (size - before - after));
        for (int i = before; i < size - after; ++i) middle.push_back(std::move(all[(std::size_t) i]));
        *out++ = make_list(std::move(middle));
        for (int i = before - 1; i >= 0; --i) *out++ = std::move(all[(std::size_t) i]);
        f->sp = out;
    }
    void call_other(Frame* f, int total, const TupleObj* kw) {
        Value* args = f->sp - total;
        const int npos = total - (kw ? (int) kw->v.size() : 0);
        Value out = call(args[-1], args, npos, kw);
        clear_range(args - 1, f->sp);
        args[-1] = std::move(out);
        f->sp = args;
    }
    // True when a frame was pushed: `self.dfs(i + 1)` stays on heap frames like any other Python call.
    bool call_method(Frame* f, const Code* code, int name_index, int total, const TupleObj* kw) {
        const std::string_view name = code->names[(std::size_t) name_index];
        Value* recv = f->sp - total - 1;
        const int npos = total - (kw ? (int) kw->v.size() : 0);
        Value out;
        const NativeDef* method = nullptr;
        const Value* in_class = nullptr;
        if (recv->tag == Tag::Instance) {
            auto* self = recv->as<InstanceObj>();
            if (!find_attr(self->attrs, name)) in_class = find_attr(self->cls.as<ClassObj>()->methods, name);
        } else if (recv->tag != Tag::Module && recv->tag != Tag::Type && recv->tag != Tag::Stub && recv->tag != Tag::Class) {
            method = method_for(type_of(*recv), name);
        }
        if (in_class && in_class->tag == Tag::Function) {
            const Value fn = *in_class;
            Value gen = enter(fn, recv, npos + 1, kw);
            clear_range(recv, f->sp);
            f->sp = recv;
            if (gen.empty()) return true;
            *f->sp++ = std::move(gen);
            return false;
        }
        if (in_class) {
            const Value fn = *in_class;
            out = call(fn, recv, npos + 1, kw);
        } else if (method) {
            if (method->is_static) out = call_native(method, recv + 1, npos, kw);
            else out = call_native(method, recv, npos + 1, kw, 1);
        } else {
            const Value attr = get_attr(*recv, name);
            out = call(attr, recv + 1, npos, kw);
        }
        clear_range(recv, f->sp);
        *recv = std::move(out);
        f->sp = recv + 1;
        return false;
    }
    // True when a frame was pushed.
    bool call_ex(Frame* f, bool has_dict) {
        Value* sp = f->sp;
        const Value callee = sp[has_dict ? -3 : -2];
        Vals args = std::move(sp[has_dict ? -2 : -1].as<ListObj>()->v);
        Value names_holder;
        const TupleObj* kw = nullptr;
        if (has_dict) {
            Vals names;
            for (const auto& e : sp[-1].as<DictObj>()->t.entries) {
                if (e.key.empty()) continue;
                if (e.key.tag != Tag::Str) raise(T_TypeError, "keywords must be strings");
                names.push_back(e.key);
            }
            for (const auto& e : sp[-1].as<DictObj>()->t.entries)
                if (!e.key.empty()) args.push_back(e.val);
            names_holder = make_tuple(std::move(names));
            kw = names_holder.as<TupleObj>();
            if (kw->v.empty()) kw = nullptr;
        }
        const int npos = (int) args.size() - (kw ? (int) kw->v.size() : 0);
        Value* slot = sp - (has_dict ? 3 : 2);
        if (callee.tag == Tag::Function) {
            Value gen = enter(callee, args.data(), npos, kw);
            clear_range(slot, sp);
            f->sp = slot;
            if (gen.empty()) return true;
            *slot = std::move(gen);
            f->sp = slot + 1;
            return false;
        }
        if (args.empty()) args.resize(1);
        Value out = call(callee, args.data(), npos, kw);
        clear_range(slot, sp);
        *slot = std::move(out);
        f->sp = slot + 1;
        return false;
    }
    void make_function(Frame* f, int index, int flags) {
        auto* fn = new FunctionObj(&compiler.codes[(std::size_t) index]);
        Value holder(fn);
        Value* sp = f->sp;
        if (flags & 4) fn->closure = seq_of(*--sp);
        if (flags & 2) fn->kw_defaults = seq_of(*--sp);
        if (flags & 1) fn->defaults = seq_of(*--sp);
        const int popped = ((flags & 1) + ((flags >> 1) & 1) + ((flags >> 2) & 1));
        clear_range(sp, sp + popped);
        *sp = std::move(holder);
        f->sp = sp + 1;
    }
    static void check_catchable(const Value& t) {
        if (t.tag != Tag::Type || !subtype((int) t.i, T_BaseException)) raise(T_TypeError, "catching classes that do not inherit from BaseException is not allowed");
    }
    static bool exception_matches(const Value& exc, const Value& pattern) {
        const int type = exc.as<ExcObj>()->type;
        if (pattern.tag != Tag::Tuple) {
            check_catchable(pattern);
            return subtype(type, (int) pattern.i);
        }
        for (const auto& item : seq_of(pattern)) check_catchable(item);
        for (const auto& item : seq_of(pattern))
            if (subtype(type, (int) item.i)) return true;
        return false;
    }
    static Value to_exception(const Value& v) {
        if (v.tag == Tag::Exc) return v;
        if (v.tag == Tag::Type && subtype((int) v.i, T_BaseException)) return Value(new ExcObj((int) v.i));
        raise(T_TypeError, "exceptions must derive from BaseException");
    }
    Value raise_value_of(Frame* f, int argc) {
        Value* sp = f->sp;
        if (argc == 0) {
            if (handled.empty()) raise(T_RuntimeError, "No active exception to reraise");
            return handled.back();
        }
        if (argc == 2) {
            const Value& cause = sp[-1];
            const bool ok = cause.is_none() || cause.tag == Tag::Exc || (cause.tag == Tag::Type && subtype((int) cause.i, T_BaseException));
            if (!ok) raise(T_TypeError, "exception causes must derive from BaseException");
        }
        Value exc = to_exception(sp[-argc]);
        clear_range(sp - argc, sp);
        f->sp = sp - argc;
        return exc;
    }
    [[noreturn]] void unbound_local(const Frame* f, int slot) {
        raise(T_UnboundLocalError, "cannot access local variable '" + std::string(f->code->slots[(std::size_t) slot]) + "' where it is not associated with a value");
    }
    [[noreturn]] void unbound_cell(const Frame* f, int slot) {
        const std::string name(f->code->slots[(std::size_t) slot]);
        if (slot >= f->code->nlocals) raise(T_NameError, "cannot access free variable '" + name + "' where it is not associated with a value in enclosing scope");
        unbound_local(f, slot);
    }
    [[noreturn]] void name_error(int slot) { raise(T_NameError, "name '" + std::string(compiler.global_names[(std::size_t) slot]) + "' is not defined"); }
    void import_from(Frame* f, const Code* code, int index) {
        const std::string_view name = code->names[(std::size_t) index];
        Value* sp = f->sp;
        *sp = get_attr(sp[-1], name);
        f->sp = sp + 1;
    }
    static std::string function_text(const Value& v) {
        if (v.tag == Tag::Function) return "__main__." + v.as<FunctionObj>()->code->qualname + "()";
        if (v.tag == Tag::Native) return std::string(v.native->name) + "()";
        if (v.tag == Tag::Type) return std::string(kTypes[v.i].name) + "()";
        return std::string(type_name(v)) + " object";
    }
    void dict_merge(Frame* f, int at, bool check_duplicates) {
        Value* sp = f->sp;
        const Value source = std::move(sp[-1]);
        f->sp = sp - 1;
        const std::string callee = check_duplicates ? function_text(f->sp[-at - 2]) : "";
        if (source.tag != Tag::Dict) {
            if (check_duplicates) raise(T_TypeError, callee + " argument after ** must be a mapping, not " + type_name(source));
            raise(T_TypeError, quoted(source) + " object is not a mapping");
        }
        auto* d = f->sp[-at].as<DictObj>();
        if (check_duplicates) {
            for (const auto& e : source.as<DictObj>()->t.entries) {
                if (e.key.empty()) continue;
                if (e.key.tag != Tag::Str) raise(T_TypeError, "keywords must be strings");
                if (d->t.get(e.key)) raise(T_TypeError, callee + " got multiple values for keyword argument " + repr_utf8(e.key));
            }
        }
        dict_update_from(d, source);
    }
    Value execute(std::size_t stop);
};

#define SYNC() (f->ip = ip, f->sp = sp)
#define RELOAD() (f = &frames[depth - 1], ip = f->ip, sp = f->sp, fast = f->base, begin = f->code->ins.data())
inline Value VM::execute(std::size_t stop) {
    Runtime* const r = rt;
    Frame* f = &frames[depth - 1];
    const Instr* ip = f->ip;
    Value* sp = f->sp;
    Value* fast = f->base;
    const Instr* begin = f->code->ins.data();
    const Value* const constants = program.constants.data();
    for (;;) {
        const Instr in = *ip++;
        if (++r->steps >= r->next_poll) {
            SYNC();
            r->poll();
        }
        switch (in.op) {
            case Op::LoadConst: *sp++ = constants[in.a]; break;
            case Op::LoadFast:
                if (fast[in.a].empty()) {
                    SYNC();
                    unbound_local(f, in.a);
                }
                *sp++ = fast[in.a];
                break;
            case Op::StoreFast: fast[in.a] = std::move(*--sp); break;
            case Op::DeleteFast:
                if (fast[in.a].empty()) {
                    SYNC();
                    unbound_local(f, in.a);
                }
                fast[in.a].reset();
                break;
            case Op::LoadDeref: {
                const auto* cell = fast[in.a].as<CellObj>();
                if (cell->v.empty()) {
                    SYNC();
                    unbound_cell(f, in.a);
                }
                *sp++ = cell->v;
                break;
            }
            case Op::StoreDeref: fast[in.a].as<CellObj>()->v = std::move(*--sp); break;
            case Op::DeleteDeref: {
                auto* cell = fast[in.a].as<CellObj>();
                if (cell->v.empty()) {
                    SYNC();
                    unbound_cell(f, in.a);
                }
                cell->v.reset();
                break;
            }
            case Op::LoadClosure: *sp++ = fast[in.a]; break;
            case Op::LoadGlobal: {
                const Value& g = globals[(std::size_t) in.a];
                if (!g.empty()) {
                    *sp++ = g;
                    break;
                }
                const Value& b = builtins[(std::size_t) in.a];
                if (b.empty()) {
                    SYNC();
                    name_error(in.a);
                }
                *sp++ = b;
                break;
            }
            case Op::StoreGlobal: globals[(std::size_t) in.a] = std::move(*--sp); break;
            case Op::DeleteGlobal:
                if (globals[(std::size_t) in.a].empty()) {
                    SYNC();
                    name_error(in.a);
                }
                globals[(std::size_t) in.a].reset();
                break;
            case Op::Pop: (--sp)->reset(); break;
            case Op::Dup: *sp = sp[-1]; ++sp; break;
            case Op::Dup2:
                sp[0] = sp[-2];
                sp[1] = sp[-1];
                sp += 2;
                break;
            case Op::Rot2: std::swap(sp[-1], sp[-2]); break;
            case Op::Rot3:
                std::swap(sp[-1], sp[-2]);
                std::swap(sp[-2], sp[-3]);
                break;
            case Op::Add: {
                Value& a = sp[-2];
                const Value& b = sp[-1];
                if (a.tag == Tag::Int && b.tag == Tag::Int) {
                    const int64_t out = (int64_t) ((uint64_t) a.i + (uint64_t) b.i);
                    if (((a.i ^ out) & (b.i ^ out)) >= 0) {
                        a.i = out;
                        --sp;
                        sp->tag = Tag::Empty;
                        break;
                    }
                } else if (a.tag == Tag::Float && b.tag == Tag::Float) {
                    a.d += b.d;
                    --sp;
                    sp->tag = Tag::Empty;
                    break;
                }
                SYNC();
                binary_top(f, Bin::Add, in.b != 0);
                sp = f->sp;
                break;
            }
            case Op::Sub: {
                Value& a = sp[-2];
                const Value& b = sp[-1];
                if (a.tag == Tag::Int && b.tag == Tag::Int) {
                    const int64_t out = (int64_t) ((uint64_t) a.i - (uint64_t) b.i);
                    if (((a.i ^ b.i) & (a.i ^ out)) >= 0) {
                        a.i = out;
                        --sp;
                        sp->tag = Tag::Empty;
                        break;
                    }
                } else if (a.tag == Tag::Float && b.tag == Tag::Float) {
                    a.d -= b.d;
                    --sp;
                    sp->tag = Tag::Empty;
                    break;
                }
                SYNC();
                binary_top(f, Bin::Sub, in.b != 0);
                sp = f->sp;
                break;
            }
            case Op::Mul:
                if (sp[-2].tag == Tag::Float && sp[-1].tag == Tag::Float) {
                    sp[-2].d *= sp[-1].d;
                    --sp;
                    sp->tag = Tag::Empty;
                    break;
                }
                SYNC();
                binary_top(f, Bin::Mul, in.b != 0);
                sp = f->sp;
                break;
            case Op::Binary:
                SYNC();
                binary_top(f, (Bin) in.a, in.b != 0);
                sp = f->sp;
                break;
            case Op::Not: {
                SYNC();
                const bool out = !truth(sp[-1]);
                sp[-1] = Value::boolean(out);
                break;
            }
            case Op::Negate: case Op::Positive: case Op::Invert:
                SYNC();
                unary_top(f, in.op);
                break;
            case Op::Compare: {
                Value& a = sp[-2];
                const Value& b = sp[-1];
                if (a.tag == Tag::Int && b.tag == Tag::Int && in.a <= (int) CmpOp::Ne) {
                    bool out = false;
                    switch ((CmpOp) in.a) {
                        case CmpOp::Lt: out = a.i < b.i; break;
                        case CmpOp::Le: out = a.i <= b.i; break;
                        case CmpOp::Gt: out = a.i > b.i; break;
                        case CmpOp::Ge: out = a.i >= b.i; break;
                        case CmpOp::Eq: out = a.i == b.i; break;
                        default: out = a.i != b.i;
                    }
                    a.tag = Tag::Bool;
                    a.b = out;
                    --sp;
                    sp->tag = Tag::Empty;
                    break;
                }
                SYNC();
                compare_top(f, (CmpOp) in.a);
                sp = f->sp;
                break;
            }
            case Op::Jump: ip = begin + in.a; break;
            case Op::JumpIfFalse: case Op::JumpIfTrue: {
                bool yes;
                Value& top_value = sp[-1];
                if (top_value.tag == Tag::Bool) {
                    yes = top_value.b;
                    top_value.tag = Tag::Empty;
                    --sp;
                } else {
                    SYNC();
                    yes = truth(top_value);
                    (--sp)->reset();
                }
                if (yes == (in.op == Op::JumpIfTrue)) ip = begin + in.a;
                break;
            }
            case Op::JumpIfFalseOrPop: case Op::JumpIfTrueOrPop: {
                SYNC();
                const bool yes = truth(sp[-1]);
                if (yes == (in.op == Op::JumpIfTrueOrPop)) ip = begin + in.a;
                else (--sp)->reset();
                break;
            }
            case Op::GetIter:
                SYNC();
                sp[-1] = get_iter(sp[-1]);
                break;
            case Op::ForIter: {
                SYNC();
                if (sp[-1].as<IterObj>()->next(*sp)) {
                    ++sp;
                } else {
                    (--sp)->reset();
                    ip = begin + in.a;
                }
                break;
            }
            case Op::BuildList: case Op::BuildTuple:
                SYNC();
                build_seq(f, in.a, in.op == Op::BuildTuple);
                sp = f->sp;
                break;
            case Op::BuildSet:
                SYNC();
                build_set(f, in.a);
                sp = f->sp;
                break;
            case Op::BuildDict: *sp++ = make_dict(); break;
            case Op::BuildSlice: {
                SYNC();
                Value s = make_slice(sp[-3], sp[-2], sp[-1]);
                clear_range(sp - 3, sp);
                sp -= 3;
                *sp++ = std::move(s);
                break;
            }
            case Op::BuildString:
                SYNC();
                build_string(f, in.a);
                sp = f->sp;
                break;
            case Op::FormatValue:
                SYNC();
                format_top(f, in.a, in.b != 0);
                sp = f->sp;
                break;
            case Op::ListAppend:
                SYNC();
                sp[-1 - in.a].as<ListObj>()->v.push_back(std::move(sp[-1]));
                --sp;
                break;
            case Op::SetAdd:
                SYNC();
                sp[-1 - in.a].as<SetObj>()->t.set(sp[-1], Value::none());
                (--sp)->reset();
                break;
            case Op::MapAdd:
                SYNC();
                sp[-2 - in.a].as<DictObj>()->t.set(sp[-2], sp[-1]);
                (--sp)->reset();
                (--sp)->reset();
                break;
            case Op::ListExtend:
                SYNC();
                if (in.b && !is_iterable(sp[-1])) raise(T_TypeError, function_text(sp[-3]) + " argument after * must be an iterable, not " + type_name(sp[-1]));
                list_extend(sp[-1 - in.a].as<ListObj>(), sp[-1]);
                (--sp)->reset();
                break;
            case Op::SetUpdate:
                SYNC();
                set_binary(Bin::Or, sp[-1 - in.a], as_set(sp[-1]), true);
                (--sp)->reset();
                break;
            case Op::DictUpdate: case Op::DictMerge:
                SYNC();
                dict_merge(f, in.a, in.op == Op::DictMerge);
                sp = f->sp;
                break;
            case Op::ListToTuple: {
                Vals v = std::move(sp[-1].as<ListObj>()->v);
                sp[-1] = make_tuple(std::move(v));
                break;
            }
            case Op::Subscr: {
                SYNC();
                Value out = subscript(sp[-2], sp[-1]);
                (--sp)->reset();
                sp[-1] = std::move(out);
                break;
            }
            case Op::StoreSubscr:
                SYNC();
                store_subscript(sp[-2], sp[-1], sp[-3]);
                clear_range(sp - 3, sp);
                sp -= 3;
                break;
            case Op::DeleteSubscr:
                SYNC();
                delete_subscript(sp[-2], sp[-1]);
                clear_range(sp - 2, sp);
                sp -= 2;
                break;
            case Op::LoadAttr:
                SYNC();
                sp[-1] = get_attr(sp[-1], f->code->names[(std::size_t) in.a]);
                break;
            case Op::KwNames: pending_kw = constants[in.a].as<TupleObj>(); break;
            case Op::Call: {
                const TupleObj* kw = pending_kw;
                pending_kw = nullptr;
                SYNC();
                Value* args = sp - in.a;
                if (args[-1].tag == Tag::Function) {
                    Value gen = enter(args[-1], args, in.a - (kw ? (int) kw->v.size() : 0), kw);
                    clear_range(args - 1, sp);
                    sp = args - 1;
                    if (gen.empty()) {
                        f->sp = sp;
                        RELOAD();
                    } else {
                        *sp++ = std::move(gen);
                    }
                    break;
                }
                call_other(f, in.a, kw);
                sp = f->sp;
                break;
            }
            case Op::CallMethod: {
                const TupleObj* kw = pending_kw;
                pending_kw = nullptr;
                SYNC();
                if (call_method(f, f->code, in.a, in.b, kw)) {
                    RELOAD();
                } else {
                    sp = f->sp;
                }
                break;
            }
            case Op::BuildClass:
                SYNC();
                *sp = Value(new ClassObj(f->code->names[(std::size_t) in.a]));
                ++sp;
                break;
            case Op::AddMethod:
                SYNC();
                sp[-2].as<ClassObj>()->methods.emplace_back(f->code->names[(std::size_t) in.a], std::move(sp[-1]));
                --sp;
                break;
            case Op::StoreAttr:
                SYNC();
                set_attr(sp[-1], f->code->names[(std::size_t) in.a], std::move(sp[-2]));
                clear_range(sp - 2, sp);
                sp -= 2;
                break;
            case Op::DeleteAttr:
                SYNC();
                delete_attr(sp[-1], f->code->names[(std::size_t) in.a]);
                sp[-1].reset();
                --sp;
                break;
            case Op::CallEx:
                SYNC();
                if (call_ex(f, in.b != 0)) {
                    RELOAD();
                } else {
                    sp = f->sp;
                }
                break;
            case Op::MakeFunction:
                SYNC();
                make_function(f, in.a, in.b);
                sp = f->sp;
                break;
            case Op::Return: {
                Value out = std::move(*--sp);
                f->sp = sp;
                leave(f);
                if (depth == stop) return out;
                RELOAD();
                *sp++ = std::move(out);
                break;
            }
            case Op::Yield: {
                Value out = std::move(*--sp);
                SYNC();
                GenObj* gen = f->gen;
                gen->saved.ip = ip;
                gen->saved.sp = sp;
                gen->saved.block = f->block;
                f->func.reset();
                --depth;
                return out;
            }
            case Op::Unpack:
                SYNC();
                unpack(f, in.a);
                sp = f->sp;
                break;
            case Op::UnpackEx:
                SYNC();
                unpack_star(f, in.a, in.b);
                sp = f->sp;
                break;
            case Op::Raise: {
                SYNC();
                Value exc = raise_value_of(f, in.a);
                if (!exc.as<ExcObj>()->line) exc.as<ExcObj>()->line = line_of(*f);
                if (!unwind(exc, stop)) throw PyError{std::move(exc)};
                RELOAD();
                break;
            }
            case Op::Reraise: {
                Value exc = std::move(*--sp);
                f->sp = sp;
                f->ip = ip;
                if (!unwind(exc, stop)) throw PyError{std::move(exc)};
                RELOAD();
                break;
            }
            case Op::SetupTry: {
                Value block = Value::of(Tag::Block);
                block.i = (int64_t) in.a << 32 | (uint32_t) (f->block + 1);
                block.aux = (uint32_t) handled.size();
                f->block = (int) (sp - fast);
                *sp++ = block;
                break;
            }
            case Op::PopBlock:
                --sp;
                f->block = (int) (uint32_t) sp->i - 1;
                sp->tag = Tag::Empty;
                break;
            case Op::PopExcept: handled.pop_back(); break;
            case Op::PushHandled:
                SYNC();
                handled.push_back(sp[-1]);
                break;
            case Op::ExcMatch: {
                SYNC();
                const bool out = exception_matches(sp[-2], sp[-1]);
                (--sp)->reset();
                sp[-1] = Value::boolean(out);
                break;
            }
            case Op::Import:
                SYNC();
                *sp = import_module(f->code->names[(std::size_t) in.a]);
                ++sp;
                break;
            case Op::ImportFrom:
                SYNC();
                import_from(f, f->code, in.a);
                sp = f->sp;
                break;
        }
    }
}
#undef SYNC
#undef RELOAD

inline bool GenObj::next(Value& out) {
    if (running) raise(T_ValueError, "generator already executing");
    if (finished) return false;
    VM& machine = *vm;
    if (machine.depth > (std::size_t) rt->limits.recursion_depth) machine.too_deep();
    Nest nest;
    const std::size_t at = machine.depth;
    Frame& f = machine.frames[machine.depth++];
    f.code = saved.code;
    f.ip = saved.ip;
    f.base = saved.base;
    f.sp = saved.sp;
    f.func = saved.func;
    f.gen = this;
    f.block = saved.block;
    f.handled = (uint32_t) machine.handled.size();
    running = true;
    const Value self(this);   // the frame's code may drop the last reference to its own generator
    struct Finish {
        GenObj* gen;
        bool completed = false;
        ~Finish() {
            gen->running = false;
            if (!completed) gen->finished = true;
        }
    } finish{this};
    out = machine.run(at);
    finish.completed = true;
    if (finished) {
        out.reset();
        return false;
    }
    return true;
}
inline Value call_value(const Value& f, Value* args, int n, const TupleObj* kw) { return vm->call(f, args, n, kw); }
inline Value make_cached_call(CachedObj* f, Value* args, int n, const TupleObj* kw) {
    Value key;
    const int nkw = kw ? (int) kw->v.size() : 0;
    if (n == 1 && !nkw && (is_int(args[0]) ? args[0].tag != Tag::Bool : args[0].tag == Tag::Str)) {
        key = args[0];
    } else {
        Vals parts;
        parts.reserve((std::size_t) (n + nkw * 2 + 1));
        for (int i = 0; i < n; ++i) parts.push_back(args[i]);
        if (nkw) parts.push_back(Value::type(T_object));
        for (int i = 0; i < nkw; ++i) {
            parts.push_back(kw->v[(std::size_t) i]);
            parts.push_back(args[n + i]);
        }
        key = make_tuple(std::move(parts));
    }
    if (f->maxsize != 0) {
        if (Value* hit = f->t.get(key)) {
            ++f->hits;
            const Value out = *hit;
            if (f->maxsize > 0) {
                f->t.erase(key);
                f->t.set(key, out);
            }
            return out;
        }
    } else {
        hash_of(key);
    }
    ++f->misses;
    const Value keep_fn = f->fn;
    Value out = vm->call(keep_fn, args, n, kw);
    if (f->maxsize == 0) return out;
    if (f->t.get(key)) return out;
    if (f->maxsize > 0 && (int64_t) f->t.used >= f->maxsize) {
        for (std::size_t i = 0; i < f->t.entries.size(); ++i) {
            rt->tick();
            if (f->t.entries[i].key.empty()) continue;
            const Value oldest = f->t.entries[i].key;
            f->t.erase(oldest);
            break;
        }
    }
    f->t.set(key, out);
    return out;
}

inline void write_output(const U32& s) {
    std::string piece;
    piece.reserve(s.size());
    for (const char32_t c : s) {
        if (c >= 0xD800 && c <= 0xDFFF) raise(T_UnicodeEncodeError, "'utf-8' codec can't encode character: surrogates not allowed");
        put_utf8(piece, c);
    }
    // Once a piece is dropped nothing follows it: a later short write would read as if nothing had been lost.
    if (rt->truncated) return;
    std::string& out = rt->output;
    const std::size_t limit = rt->limits.output_bytes;
    if (out.size() + piece.size() <= limit) {
        out += piece;
        return;
    }
    std::size_t room = out.size() < limit ? limit - out.size() : 0;
    while (room > 0 && ((unsigned char) piece[room] & 0xC0) == 0x80) --room;
    out.append(piece, 0, room);
    rt->truncated = true;
}
inline Value b_print(Call& c) {
    only_keywords(c, {"sep", "end", "file", "flush"}, "print");
    Value* file = c.keyword("file");
    if (file && !file->is_none()) refuse("print to a file");
    Value* sep = c.keyword("sep");
    Value* end = c.keyword("end");
    if (sep && !sep->is_none() && sep->tag != Tag::Str) raise(T_TypeError, std::string("sep must be None or a string, not ") + type_name(*sep));
    if (end && !end->is_none() && end->tag != Tag::Str) raise(T_TypeError, std::string("end must be None or a string, not ") + type_name(*end));
    for (int i = 0; i < c.n; ++i) {
        if (i) {
            if (sep && sep->tag == Tag::Str) write_output(str_of(*sep));
            else write_output(U" ");
        }
        write_output(str_of(str_value(c.a[i])));
    }
    if (end && end->tag == Tag::Str) write_output(str_of(*end));
    else write_output(U"\n");
    return Value::none();
}
inline Value b_len(Call& c) { return Value::integer((int64_t) len_of(c.a[0])); }
inline Value b_abs(Call& c) {
    const Value& v = c.a[0];
    if (is_int(v)) return int_abs(v);
    if (v.tag == Tag::Float) return Value::real(std::fabs(v.d));
    if (v.tag == Tag::Fraction) return frac_make(int_abs(v.as<FractionObj>()->num), v.as<FractionObj>()->den);
    raise(T_TypeError, "bad operand type for abs(): " + quoted(v));
}
inline Value b_sum(Call& c) {
    only_keywords(c, {"start"}, "sum");
    Value* kw = c.keyword("start");
    Value total = kw ? *kw : c.n > 1 ? c.a[1] : Value::integer(0);
    if (total.tag == Tag::Str) raise(T_TypeError, "sum() can't sum strings [use ''.join(seq) instead]");
    each(c.a[0], [&](Value& x) { total = binary(Bin::Add, total, x, false); });
    return total;
}
inline Value extreme(Call& c, const char* name, Cmp op) {
    only_keywords(c, {"key", "default"}, name);
    Value* key_kw = c.keyword("key");
    Value* fallback = c.keyword("default");
    const Value key = key_kw ? *key_kw : Value::none();
    if (c.n > 1 && fallback) raise(T_TypeError, std::string("Cannot specify a default for ") + name + "() with multiple positional arguments");
    Value best, best_key;
    auto consider = [&](const Value& item) {
        Value k = item;
        if (!key.is_none()) {
            Value arg = item;
            k = call_value(key, &arg, 1);
        }
        if (best.empty() || order(op, k, best_key)) {
            best = item;
            best_key = std::move(k);
        }
    };
    if (c.n > 1) {
        for (int i = 0; i < c.n; ++i) {
            rt->tick();
            consider(c.a[i]);
        }
    } else {
        each(c.a[0], [&](Value& x) { consider(x); });
    }
    if (best.empty()) {
        if (fallback) return *fallback;
        raise(T_ValueError, std::string(name) + "() arg is an empty sequence");
    }
    return best;
}
inline Value b_min(Call& c) { return extreme(c, "min", Cmp::Lt); }
inline Value b_max(Call& c) { return extreme(c, "max", Cmp::Gt); }
inline Value b_sorted(Call& c) {
    only_keywords(c, {"key", "reverse"}, "sort");
    Vals items = to_vals(c.a[0]);
    Value* key = c.keyword("key");
    Value* rev = c.keyword("reverse");
    sort_values(items, key ? *key : Value::none(), rev ? truth(*rev) : false);
    return make_list(std::move(items));
}
inline Value b_any(Call& c) {
    Value it = get_iter(c.a[0]), x;
    while (iter_next(it, x)) {
        rt->tick();
        if (truth(x)) return Value::boolean(true);
    }
    return Value::boolean(false);
}
inline Value b_all(Call& c) {
    Value it = get_iter(c.a[0]), x;
    while (iter_next(it, x)) {
        rt->tick();
        if (!truth(x)) return Value::boolean(false);
    }
    return Value::boolean(true);
}
inline Value b_repr(Call& c) {
    U32 out;
    repr_to(out, c.a[0]);
    return text(std::move(out));
}
inline Value b_ascii(Call& c) {
    U32 out;
    repr_to(out, c.a[0]);
    need_ascii(out, "ascii()");
    return text(std::move(out));
}
inline Value round_half_even(const Value& n, const Value& d) {
    Value q, r;
    int_divmod(n, d, &q, &r);
    const int c = int_cmp(int_add(r, r), d);
    if (c > 0 || (c == 0 && !int_is_zero(int_mod(q, Value::integer(2))))) q = int_add(q, Value::integer(1));
    return q;
}
inline Value b_round(Call& c) {
    only_keywords(c, {"number", "ndigits"}, "round");
    Value* kw = c.keyword("ndigits");
    Value* number = c.keyword("number");
    if (!c.n && !number) raise(T_TypeError, "round() missing required argument 'number' (pos 1)");
    const Value digits = kw ? *kw : c.n > 1 ? c.a[1] : Value::none();
    const Value v = number ? *number : c.a[0];
    if (v.tag == Tag::Float) {
        if (digits.is_none()) {
            if (std::isnan(v.d)) raise(T_ValueError, "cannot convert float NaN to integer");
            if (std::isinf(v.d)) raise(T_OverflowError, "cannot convert float infinity to integer");
            double rounded = std::round(v.d);
            if (std::fabs(v.d - rounded) == 0.5) rounded = 2.0 * std::round(v.d / 2.0);
            return int_from_double(rounded);
        }
        const int64_t n = int_arg(digits);
        if (n > 323) return v;
        if (n < -308) return Value::real(0.0 * v.d);
        return Value::real(float_round(v.d, n));
    }
    if (is_int(v)) {
        const Value self = v.tag == Tag::Bool ? Value::integer(small_of(v)) : v;
        if (digits.is_none()) return self;
        const int64_t n = int_arg(digits);
        if (n >= 0) return self;
        if (n < -100000) too_big();
        const Value p = int_pow(Value::integer(10), (uint64_t) -n);
        return int_mul(round_half_even(self, p), p);
    }
    if (v.tag == Tag::Fraction) {
        auto* f = v.as<FractionObj>();
        if (digits.is_none()) return round_half_even(f->num, f->den);
        const int64_t n = int_arg(digits);
        const Value shift = int_pow(Value::integer(10), (uint64_t) (n < 0 ? -n : n));
        if (n > 0) return frac_make(round_half_even(int_mul(f->num, shift), f->den), shift);
        return frac_make(int_mul(round_half_even(f->num, int_mul(f->den, shift)), shift), Value::integer(1));
    }
    raise(T_TypeError, std::string("type ") + type_name(v) + " doesn't define __round__ method");
}
inline Value int_powmod(const Value& base_in, const Value& exp_in, const Value& mod) {
    if (int_is_zero(mod)) raise(T_ValueError, "pow() 3rd argument cannot be 0");
    Value base = int_mod(base_in, mod), exponent = exp_in;
    if (int_is_neg(exponent)) {
        // The inverse of base, by the extended Euclid algorithm over the modulus.
        Value a = base, b = mod, x0 = Value::integer(1), x1 = Value::integer(0);
        while (!int_is_zero(b)) {
            rt->tick();
            Value q, r;
            int_divmod(a, b, &q, &r);
            Value next = int_sub(x0, int_mul(q, x1));
            a = std::move(b);
            b = std::move(r);
            x0 = std::move(x1);
            x1 = std::move(next);
        }
        if (!eq(int_abs(a), Value::integer(1))) raise(T_ValueError, "base is not invertible for the given modulus");
        base = int_mod(int_is_neg(a) ? int_negate(x0) : x0, mod);
        exponent = int_negate(exponent);
    }
    Value out = int_mod(Value::integer(1), mod);
    const std::size_t bits = int_bits(exponent);
    for (std::size_t i = bits; i-- > 0;) {
        out = int_mod(int_mul(out, out), mod);
        const IntRef e(exponent);
        if ((e.p[i / 32] >> (i % 32)) & 1) out = int_mod(int_mul(out, base), mod);
    }
    return out;
}
inline Value b_pow(Call& c) {
    only_keywords(c, {"base", "exp", "mod"}, "pow");
    Value* kw_base = c.keyword("base");
    Value* kw_exp = c.keyword("exp");
    Value* kw_mod = c.keyword("mod");
    if (c.n + c.nkw() < 2) raise(T_TypeError, "pow() missing required argument 'exp' (pos 2)");
    const Value base = kw_base ? *kw_base : c.a[0];
    const Value exponent = kw_exp ? *kw_exp : c.a[1];
    const Value mod = kw_mod ? *kw_mod : c.n > 2 ? c.a[2] : Value::none();
    if (mod.is_none()) return binary(Bin::Pow, base, exponent, false);
    if (!is_int(base) || !is_int(exponent) || !is_int(mod)) raise(T_TypeError, "pow() 3rd argument not allowed unless all arguments are integers");
    return int_powmod(base, exponent, mod);
}
inline Value b_divmod(Call& c) {
    const Value& a = c.a[0];
    const Value& b = c.a[1];
    if (is_int(a) && is_int(b)) {
        if (int_is_zero(b)) raise(T_ZeroDivisionError, "integer division or modulo by zero");
        Value q, r;
        int_divmod(a, b, &q, &r);
        return pair(std::move(q), std::move(r));
    }
    if (is_number(a) && is_number(b)) {
        const double x = to_float(a), y = to_float(b);
        if (y == 0) raise(T_ZeroDivisionError, "float divmod()");
        double q = 0, r = 0;
        float_divmod(x, y, q, r);
        return pair(Value::real(q), Value::real(r));
    }
    if (is_rational(a) && is_rational(b)) return pair(frac_binary(Bin::FloorDiv, a, b), frac_binary(Bin::Mod, a, b));
    raise(T_TypeError, "unsupported operand type(s) for divmod(): " + quoted(a) + " and " + quoted(b));
}
inline Value b_ord(Call& c) {
    const Value& v = c.a[0];
    if (v.tag != Tag::Str) raise(T_TypeError, std::string("ord() expected string of length 1, but ") + type_name(v) + " found");
    if (str_of(v).size() != 1) raise(T_TypeError, "ord() expected a character, but string of length " + std::to_string(str_of(v).size()) + " found");
    return Value::integer((int64_t) str_of(v)[0]);
}
inline Value b_chr(Call& c) {
    const int64_t code = int_arg(c.a[0]);
    if (code < 0 || code > 0x10FFFF) raise(T_ValueError, "chr() arg not in range(0x110000)");
    return text(U32(1, (char32_t) code));
}
inline Value base_text(const Value& v, int base, const char* prefix) {
    if (!is_int(v)) raise(T_TypeError, quoted(v) + " object cannot be interpreted as an integer");
    return text((int_is_neg(v) ? "-" : "") + std::string(prefix) + int_base(v, base));
}
inline Value b_hex(Call& c) { return base_text(c.a[0], 16, "0x"); }
inline Value b_oct(Call& c) { return base_text(c.a[0], 8, "0o"); }
inline Value b_bin(Call& c) { return base_text(c.a[0], 2, "0b"); }
inline bool is_instance(const Value& v, const Value& t) {
    if (t.tag == Tag::Tuple) {
        for (const auto& item : seq_of(t)) {
            rt->tick();
            if (is_instance(v, item)) return true;
        }
        return false;
    }
    if (t.tag == Tag::Stub) refuse("isinstance with a typing object");
    if (t.tag == Tag::Class) return v.tag == Tag::Instance && v.as<InstanceObj>()->cls.o == t.o;
    if (t.tag != Tag::Type) raise(T_TypeError, "isinstance() arg 2 must be a type, a tuple of types, or a union");
    return subtype(type_of(v), (int) t.i);
}
inline Value b_isinstance(Call& c) { return Value::boolean(is_instance(c.a[0], c.a[1])); }
inline Value b_issubclass(Call& c) {
    const bool own = c.a[0].tag == Tag::Class;
    if (c.a[0].tag != Tag::Type && !own) raise(T_TypeError, "issubclass() arg 1 must be a class");
    const auto test = [&](const Value& t) {
        if (t.tag == Tag::Class) return own && t.o == c.a[0].o;
        if (t.tag != Tag::Type) raise(T_TypeError, "issubclass() arg 2 must be a class, a tuple of classes, or a union");
        return own ? t.i == T_object : subtype((int) c.a[0].i, (int) t.i);
    };
    if (c.a[1].tag == Tag::Tuple) {
        for (const auto& item : seq_of(c.a[1]))
            if (test(item)) return Value::boolean(true);
        return Value::boolean(false);
    }
    return Value::boolean(test(c.a[1]));
}
inline Value b_callable(Call& c) {
    const Tag t = c.a[0].tag;
    return Value::boolean(t == Tag::Function || t == Tag::Native || t == Tag::Type || t == Tag::Bound || t == Tag::Cached || t == Tag::Class || t == Tag::Method);
}
// These hash here, but not to CPython's number. A str has no fixed number there, so any value is true of it.
inline void require_cpython_hash(const Value& v) {
    if (v.tag == Tag::Set || v.tag == Tag::Range || v.tag == Tag::Ellipsis) refuse(std::string("the hash of a ") + type_name(v));
    if (v.tag == Tag::Tuple) {
        Nest nest;
        for (const auto& item : seq_of(v)) require_cpython_hash(item);
    }
}
inline Value b_hash(Call& c) {
    require_cpython_hash(c.a[0]);
    return Value::integer((int64_t) hash_of(c.a[0]));
}
inline Value b_iter(Call& c) {
    if (c.n > 1) refuse("iter with a sentinel");
    return get_iter(c.a[0]);
}
inline Value b_next(Call& c) {
    if (c.a[0].tag != Tag::Iter) raise(T_TypeError, quoted(c.a[0]) + " object is not an iterator");
    Value out;
    if (iter_next(c.a[0], out)) return out;
    if (c.n > 1) return c.a[1];
    auto* e = new ExcObj(T_StopIteration);
    throw PyError{Value(e)};
}
inline Value b_format(Call& c) {
    if (c.n > 1 && c.a[1].tag != Tag::Str) raise(T_TypeError, std::string("format() argument 2 must be str, not ") + type_name(c.a[1]));
    return format_value(c.a[0], c.n > 1 ? str_of(c.a[1]) : U32{});
}

struct MapIter : IterObj {
    Value fn;
    Vals iters;
    MapIter() : IterObj(T_map) {}
    void clear() override { fn.reset(); iters.clear(); }
    bool next(Value& out) override {
        Nest nest;
        Vals args(iters.size());
        for (std::size_t i = 0; i < iters.size(); ++i)
            if (!iter_next(iters[i], args[i])) return false;
        out = call_value(fn, args.data(), (int) args.size());
        return true;
    }
};
struct FilterIter : IterObj {
    Value fn, it;
    FilterIter() : IterObj(T_filter) {}
    void clear() override { fn.reset(); it.reset(); }
    bool next(Value& out) override {
        Nest nest;
        Value x;
        while (iter_next(it, x)) {
            rt->tick();
            bool keep = false;
            if (fn.is_none() || (fn.tag == Tag::Type && fn.i == T_bool)) {
                keep = truth(x);
            } else {
                Value arg = x;
                keep = truth(call_value(fn, &arg, 1));
            }
            if (keep) {
                out = std::move(x);
                return true;
            }
        }
        return false;
    }
};
struct ZipIter : IterObj {
    Vals iters;
    bool strict = false;
    ZipIter() : IterObj(T_zip) {}
    void clear() override { iters.clear(); }
    bool next(Value& out) override {
        Nest nest;
        if (iters.empty()) return false;
        Vals row;
        row.reserve(iters.size());
        for (std::size_t i = 0; i < iters.size(); ++i) {
            Value x;
            if (!iter_next(iters[i], x)) {
                if (strict) strict_failure(i);
                return false;
            }
            row.push_back(std::move(x));
        }
        out = make_tuple(std::move(row));
        return true;
    }
    void strict_failure(std::size_t i) {
        if (i > 0) raise(T_ValueError, "zip() argument " + std::to_string(i + 1) + " is shorter than argument" + (i == 1 ? " 1" : "s 1-" + std::to_string(i)));
        for (std::size_t k = 1; k < iters.size(); ++k) {
            Value x;
            if (iter_next(iters[k], x)) raise(T_ValueError, "zip() argument " + std::to_string(k + 1) + " is longer than argument" + (k == 1 ? " 1" : "s 1-" + std::to_string(k)));
        }
    }
};
struct EnumerateIter : IterObj {
    Value it, n;
    EnumerateIter() : IterObj(T_enumerate) {}
    void clear() override { it.reset(); n.reset(); }
    bool next(Value& out) override {
        Nest nest;
        Value x;
        if (!iter_next(it, x)) return false;
        out = pair(n, std::move(x));
        n = int_add(n, Value::integer(1));
        return true;
    }
};
struct ChainIter : IterObj {
    Value outer, inner;
    ChainIter() : IterObj(T_chain) {}
    void clear() override { outer.reset(); inner.reset(); }
    bool next(Value& out) override {
        Nest nest;
        for (;;) {
            rt->tick();
            if (inner.empty()) {
                Value next_iterable;
                if (!iter_next(outer, next_iterable)) return false;
                inner = get_iter(next_iterable);
            }
            if (iter_next(inner, out)) return true;
            inner.reset();
        }
    }
};
struct CountIter : IterObj {
    Value at, step;
    CountIter() : IterObj(T_count) {}
    void clear() override { at.reset(); step.reset(); }
    bool next(Value& out) override {
        out = at;
        at = binary(Bin::Add, at, step, false);
        return true;
    }
};
struct IsliceIter : IterObj {
    Value it;
    int64_t next_index = 0, stop = -1, step = 1, count = 0;
    IsliceIter() : IterObj(T_islice) {}
    void clear() override { it.reset(); }
    bool next(Value& out) override {
        Nest nest;
        if (it.empty()) return false;
        Value skipped;
        while (count < next_index) {
            rt->tick();
            if (!iter_next(it, skipped)) {
                it.reset();
                return false;
            }
            ++count;
        }
        if ((stop != -1 && count >= stop) || !iter_next(it, out)) {
            it.reset();
            return false;
        }
        ++count;
        const int64_t previous = next_index;
        next_index += step;
        if (next_index < previous || (stop != -1 && next_index > stop)) next_index = stop;
        return true;
    }
};
struct AccumulateIter : IterObj {
    Value it, fn, total, initial;
    AccumulateIter() : IterObj(T_accumulate) {}
    void clear() override { it.reset(); fn.reset(); total.reset(); initial.reset(); }
    bool next(Value& out) override {
        Nest nest;
        if (!initial.empty()) {
            total = std::move(initial);
            initial.reset();
            out = total;
            return true;
        }
        Value x;
        if (!iter_next(it, x)) return false;
        if (total.empty()) {
            total = std::move(x);
        } else if (fn.is_none()) {
            total = binary(Bin::Add, total, x, false);
        } else {
            Value args[2] = {total, std::move(x)};
            total = call_value(fn, args, 2);
        }
        out = total;
        return true;
    }
};
inline std::vector<int64_t, Alloc<int64_t>> iota(int64_t n) {
    std::vector<int64_t, Alloc<int64_t>> out((std::size_t) n);
    for (int64_t i = 0; i < n; ++i) out[(std::size_t) i] = i;
    return out;
}
struct PermutationsIter : IterObj {
    Vals pool;
    std::vector<int64_t, Alloc<int64_t>> indices, cycles;
    int64_t r = 0;
    bool started = false, done = false;
    PermutationsIter() : IterObj(T_permutations) {}
    void clear() override { pool.clear(); }
    bool next(Value& out) override {
        const int64_t n = (int64_t) pool.size();
        if (done) return false;
        if (!started) {
            started = true;
            if (r > n) {
                done = true;
                return false;
            }
        } else {
            int64_t i = r - 1;
            for (; i >= 0; --i) {
                rt->tick();
                auto& c = cycles[(std::size_t) i];
                if (--c == 0) {
                    const int64_t first = indices[(std::size_t) i];
                    for (int64_t j = i; j + 1 < n; ++j) indices[(std::size_t) j] = indices[(std::size_t) j + 1];
                    indices[(std::size_t) n - 1] = first;
                    c = n - i;
                } else {
                    std::swap(indices[(std::size_t) i], indices[(std::size_t) (n - c)]);
                    break;
                }
            }
            if (i < 0) {
                done = true;
                return false;
            }
        }
        Vals row;
        row.reserve((std::size_t) r);
        rt->tick((uint64_t) r >> 3);
        for (int64_t i = 0; i < r; ++i) row.push_back(pool[(std::size_t) indices[(std::size_t) i]]);
        out = make_tuple(std::move(row));
        return true;
    }
};
struct CombinationsIter : IterObj {
    Vals pool;
    std::vector<int64_t, Alloc<int64_t>> indices;
    int64_t r = 0;
    bool repeat = false, started = false, done = false;
    explicit CombinationsIter(bool with_replacement) : IterObj(with_replacement ? T_combinations_with_replacement : T_combinations), repeat(with_replacement) {}
    void clear() override { pool.clear(); }
    bool next(Value& out) override {
        const int64_t n = (int64_t) pool.size();
        if (done) return false;
        if (!started) {
            started = true;
            if (repeat ? (n == 0 && r > 0) : r > n) {
                done = true;
                return false;
            }
            indices.assign((std::size_t) r, 0);
            if (!repeat)
                for (int64_t i = 0; i < r; ++i) indices[(std::size_t) i] = i;
        } else {
            int64_t i = r - 1;
            for (; i >= 0; --i) {
                rt->tick();
                if (indices[(std::size_t) i] != (repeat ? n - 1 : i + n - r)) break;
            }
            if (i < 0) {
                done = true;
                return false;
            }
            const int64_t base = indices[(std::size_t) i] + 1;
            for (int64_t j = i; j < r; ++j) indices[(std::size_t) j] = repeat ? base : base + (j - i);
        }
        Vals row;
        row.reserve((std::size_t) r);
        rt->tick((uint64_t) r >> 3);
        for (int64_t i = 0; i < r; ++i) row.push_back(pool[(std::size_t) indices[(std::size_t) i]]);
        out = make_tuple(std::move(row));
        return true;
    }
};
struct ProductIter : IterObj {
    std::vector<Vals, Alloc<Vals>> pools;
    std::vector<std::size_t, Alloc<std::size_t>> indices;
    bool started = false, done = false;
    ProductIter() : IterObj(T_product) {}
    void clear() override { pools.clear(); }
    bool next(Value& out) override {
        if (done) return false;
        if (!started) {
            started = true;
            for (const auto& pool : pools) {
                if (pool.empty()) {
                    done = true;
                    return false;
                }
            }
            indices.assign(pools.size(), 0);
        } else {
            std::size_t i = pools.size();
            while (i-- > 0) {
                rt->tick();
                if (++indices[i] < pools[i].size()) break;
                indices[i] = 0;
                if (i == 0) {
                    done = true;
                    return false;
                }
            }
            if (pools.empty()) {
                done = true;
                return false;
            }
        }
        Vals row;
        row.reserve(pools.size());
        rt->tick(pools.size() >> 3);
        for (std::size_t i = 0; i < pools.size(); ++i) row.push_back(pools[i][indices[i]]);
        out = make_tuple(std::move(row));
        return true;
    }
};
inline Value b_reversed_of(const Value& v) {
    switch (v.tag) {
        case Tag::List: case Tag::Tuple: return Value(new ReverseListIter(v, (int64_t) seq_of(v).size() - 1));
        case Tag::Str: return Value(new StrIter(v, -1));
        case Tag::Range: {
            auto* r = v.as<RangeObj>();
            return Value(new RangeIter(r->len ? r->start + (r->len - 1) * r->step : 0, -r->step, r->len));
        }
        case Tag::Deque: return Value(new DequeIter(v, -1));
        case Tag::Dict: case Tag::View: {
            Vals items = to_vals(v);
            std::reverse(items.begin(), items.end());
            return get_iter(make_list(std::move(items)));
        }
        default: raise(T_TypeError, quoted(v) + " object is not reversible");
    }
}
inline Value parse_int_text(const U32& raw, int64_t base) {
    if (base != 0 && (base < 2 || base > 36)) raise(T_ValueError, "int() base must be >= 2 and <= 36, or 0");
    auto bad = [&]() { raise(T_ValueError, "invalid literal for int() with base " + std::to_string(base) + ": " + repr_utf8(text(U32(raw)))); };
    std::size_t a = 0, b = raw.size();
    while (a < b && is_space(raw[a])) ++a;
    while (b > a && is_space(raw[b - 1])) --b;
    for (std::size_t i = a; i < b; ++i)
        if (raw[i] >= 128) refuse("int() of non-ASCII digits");
    bool neg = false;
    if (a < b && (raw[a] == '+' || raw[a] == '-')) neg = raw[a++] == '-';
    int use = (int) base;
    bool prefixed = false;
    if (b - a >= 2 && raw[a] == '0') {
        const char32_t p = raw[a + 1] | 32;
        const int implied = p == 'x' ? 16 : p == 'o' ? 8 : p == 'b' ? 2 : 0;
        if (implied && (base == 0 || base == implied)) {
            use = implied;
            a += 2;
            prefixed = true;
        }
    }
    if (use == 0) use = 10;
    std::string digits;
    bool last_underscore = !prefixed;
    if (a >= b) bad();
    for (std::size_t i = a; i < b; ++i) {
        const char32_t c = raw[i];
        if (c == '_') {
            if (last_underscore) bad();
            last_underscore = true;
            continue;
        }
        const int d = ascii_digit(c) ? (int) (c - '0') : ascii_alpha(c) ? (int) ((c | 32) - 'a') + 10 : 99;
        if (d >= use) bad();
        digits.push_back((char) c);
        last_underscore = false;
    }
    if (last_underscore) bad();
    if (base == 0 && !prefixed && digits.size() > 1 && digits[0] == '0' && digits.find_first_not_of('0') != std::string::npos) bad();
    return int_parse(digits, use, neg);
}
inline Value to_int(Value* a, int n, const TupleObj* kw) {
    const Value* base_kw = nullptr;
    if (kw) {
        Call c{a, n, kw};
        only_keywords(c, {"base"}, "int");
        base_kw = c.keyword("base");
    }
    if (n == 0) {
        if (base_kw) raise(T_TypeError, "int() missing string argument");
        return Value::integer(0);
    }
    if (n > 2) raise(T_TypeError, "int() takes at most 2 arguments (" + std::to_string(n) + " given)");
    const Value& v = a[0];
    if (n == 2 || base_kw) {
        if (v.tag != Tag::Str) raise(T_TypeError, "int() can't convert non-string with explicit base");
        return parse_int_text(str_of(v), int_arg(base_kw ? *base_kw : a[1]));
    }
    if (is_int(v)) return v.tag == Tag::Bool ? Value::integer(small_of(v)) : v;
    if (v.tag == Tag::Float) {
        if (std::isinf(v.d)) raise(T_OverflowError, "cannot convert float infinity to integer");
        if (std::isnan(v.d)) raise(T_ValueError, "cannot convert float NaN to integer");
        return int_from_double(std::trunc(v.d));
    }
    if (v.tag == Tag::Str) return parse_int_text(str_of(v), 10);
    if (v.tag == Tag::Fraction) {
        auto* f = v.as<FractionObj>();
        if (int_is_neg(f->num)) return int_negate(int_floordiv(int_negate(f->num), f->den));
        return int_floordiv(f->num, f->den);
    }
    raise(T_TypeError, std::string("int() argument must be a string, a bytes-like object or a real number, not '") + type_name(v) + "'");
}
inline Value to_float_value(const Value& v) {
    if (is_number(v) || v.tag == Tag::Fraction) return Value::real(to_float(v));
    if (v.tag != Tag::Str) raise(T_TypeError, std::string("float() argument must be a string or a real number, not '") + type_name(v) + "'");
    const U32& raw = str_of(v);
    auto bad = [&]() { raise(T_ValueError, "could not convert string to float: " + repr_utf8(v)); };
    std::size_t a = 0, b = raw.size();
    while (a < b && is_space(raw[a])) ++a;
    while (b > a && is_space(raw[b - 1])) --b;
    for (std::size_t i = a; i < b; ++i)
        if (raw[i] >= 128) refuse("float() of non-ASCII digits");
    std::string s;
    for (std::size_t i = a; i < b; ++i) s.push_back((char) to_lower(raw[i]));
    std::size_t i = 0;
    bool neg = false;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) neg = s[i++] == '-';
    const std::string_view rest = std::string_view(s).substr(i);
    if (rest == "inf" || rest == "infinity") return Value::real(neg ? -HUGE_VAL : HUGE_VAL);
    if (rest == "nan") return Value::real(neg ? -std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::quiet_NaN());
    std::string clean;
    std::size_t digits = 0;
    auto run = [&]() {
        bool any = false;
        while (i < s.size() && (std::isdigit((unsigned char) s[i]) || (s[i] == '_' && any && i + 1 < s.size() && std::isdigit((unsigned char) s[i + 1])))) {
            if (s[i] != '_') {
                clean.push_back(s[i]);
                ++digits;
            }
            any = true;
            ++i;
        }
    };
    run();
    if (i < s.size() && s[i] == '.') {
        clean.push_back('.');
        ++i;
        run();
    }
    if (!digits) bad();
    if (i < s.size() && s[i] == 'e') {
        clean.push_back('e');
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) clean.push_back(s[i++]);
        const std::size_t before = digits;
        run();
        if (digits == before) bad();
    }
    if (i != s.size()) bad();
    if (clean[0] == '.') clean.insert(clean.begin(), '0');
    const double out = parse_float_ascii(clean);
    return Value::real(neg ? -out : out);
}
inline Value make_range(Value* a, int n) {
    if (n < 1) raise(T_TypeError, "range expected at least 1 argument, got 0");
    if (n > 3) raise(T_TypeError, "range expected at most 3 arguments, got " + std::to_string(n));
    auto arg = [&](int i) -> int64_t {
        if (!is_int(a[i])) raise(T_TypeError, quoted(a[i]) + " object cannot be interpreted as an integer");
        if (!is_small(a[i])) refuse("range beyond 64 bits");
        return small_of(a[i]);
    };
    const int64_t start = n == 1 ? 0 : arg(0), stop = n == 1 ? arg(0) : arg(1), step = n == 3 ? arg(2) : 1;
    if (step == 0) raise(T_ValueError, "range() arg 3 must not be zero");
    constexpr int64_t edge = 1ll << 61;   // keeps every sum and length of the range inside 64 bits
    if (start > edge || start < -edge || stop > edge || stop < -edge || step > edge || step < -edge) refuse("range beyond 62 bits");
    int64_t len = 0;
    if (step > 0 && start < stop) len = (int64_t) (((uint64_t) stop - (uint64_t) start - 1) / (uint64_t) step) + 1;
    if (step < 0 && start > stop) len = (int64_t) (((uint64_t) start - (uint64_t) stop - 1) / (0 - (uint64_t) step)) + 1;
    return Value(new RangeObj(start, stop, step, len));
}
inline void require_callable_or_none(const Value& v) {
    const Tag t = v.tag;
    if (t != Tag::None && t != Tag::Function && t != Tag::Native && t != Tag::Type && t != Tag::Bound && t != Tag::Cached && t != Tag::Class && t != Tag::Method) raise(T_TypeError, "first argument must be callable or None");
}
inline Value make_fraction(Value* a, int n, const TupleObj* kw) {
    Value* den_kw = nullptr;
    Value* num_kw = nullptr;
    if (kw) {
        Call c{a, n, kw};
        num_kw = c.keyword("numerator");
        den_kw = c.keyword("denominator");
    }
    const Value numerator = n > 0 ? a[0] : num_kw ? *num_kw : Value::integer(0);
    const Value denominator = n > 1 ? a[1] : den_kw ? *den_kw : Value();
    if (denominator.empty() || denominator.is_none()) {
        if (is_int(numerator)) return frac_make(numerator, Value::integer(1));
        if (numerator.tag == Tag::Fraction) return numerator;
        if (numerator.tag == Tag::Float) return frac_from_double(numerator.d);
        if (numerator.tag == Tag::Str) return frac_from_text(str_of(numerator));
        raise(T_TypeError, "argument should be a string or a Rational instance");
    }
    if (is_rational(numerator) && is_rational(denominator)) {
        if (is_int(numerator) && is_int(denominator)) return frac_make(numerator, denominator);
        return frac_binary(Bin::Div, numerator.tag == Tag::Fraction ? numerator : frac_make(numerator, Value::integer(1)), denominator);
    }
    raise(T_TypeError, "both arguments should be Rational instances");
}
inline Value construct(int type, Value* a, int n, const TupleObj* kw) {
    const int nkw = kw ? (int) kw->v.size() : 0;
    Call c{a, n, kw};
    if (subtype(type, T_BaseException)) {
        if (nkw) raise(T_TypeError, std::string(kTypes[type].name) + "() takes no keyword arguments");
        auto* e = new ExcObj(type);
        Value holder(e);
        for (int i = 0; i < n; ++i) e->args.push_back(a[i]);
        return holder;
    }
    auto no_keywords = [&]() {
        if (nkw) raise(T_TypeError, std::string(kTypes[type].name) + "() takes no keyword arguments");
    };
    auto at_most = [&](int m) {
        if (n > m) raise(T_TypeError, std::string(kTypes[type].name) + " expected at most " + std::to_string(m) + " arguments, got " + std::to_string(n));
    };
    switch (type) {
        case T_int: return to_int(a, n, kw);
        case T_float:
            no_keywords();
            at_most(1);
            return n ? to_float_value(a[0]) : Value::real(0.0);
        case T_str:
            if (nkw || n > 1) refuse("str() with an encoding");
            return n ? str_value(a[0]) : text(U32{});
        case T_bool:
            no_keywords();
            at_most(1);
            return Value::boolean(n ? truth(a[0]) : false);
        case T_list: {
            no_keywords();
            at_most(1);
            return make_list(n ? to_vals(a[0]) : Vals{});
        }
        case T_tuple: {
            no_keywords();
            at_most(1);
            if (n && a[0].tag == Tag::Tuple) return a[0];
            return make_tuple(n ? to_vals(a[0]) : Vals{});
        }
        case T_dict: case T_Counter: {
            at_most(1);
            Value out = make_dict(type == T_Counter ? DictKind::Counter : DictKind::Plain);
            auto* d = out.as<DictObj>();
            if (n) {
                if (type == T_Counter) counter_add(d, a[0], 1);
                else dict_update_from(d, a[0]);
            }
            for (int k = 0; k < nkw; ++k) {
                if (type == T_Counter) {
                    Value* have = d->t.get(kw->v[(std::size_t) k]);
                    d->t.set(kw->v[(std::size_t) k], binary(Bin::Add, have ? *have : Value::integer(0), a[n + k], false));
                } else {
                    d->t.set(kw->v[(std::size_t) k], a[n + k]);
                }
            }
            return out;
        }
        case T_defaultdict: {
            Value out = make_dict(DictKind::Default);
            auto* d = out.as<DictObj>();
            d->factory = Value::none();
            if (n) {
                require_callable_or_none(a[0]);
                d->factory = a[0];
            }
            if (n > 1) dict_update_from(d, a[1]);
            for (int k = 0; k < nkw; ++k) d->t.set(kw->v[(std::size_t) k], a[n + k]);
            return out;
        }
        case T_set: case T_frozenset: {
            no_keywords();
            at_most(1);
            if (n && a[0].tag == Tag::Set && type == T_frozenset && a[0].as<SetObj>()->frozen) return a[0];
            Value out = make_set(type == T_frozenset);
            if (n) each(a[0], [&](Value& x) { out.as<SetObj>()->t.set(x, Value::none()); });
            return out;
        }
        case T_range: no_keywords(); return make_range(a, n);
        case T_slice: {
            no_keywords();
            at_most(3);
            if (n < 1) raise(T_TypeError, "slice expected at least 1 argument, got 0");
            if (n == 1) return make_slice(Value::none(), a[0], Value::none());
            return make_slice(a[0], a[1], n > 2 ? a[2] : Value::none());
        }
        case T_map: {
            no_keywords();
            if (n < 2) raise(T_TypeError, "map() must have at least two arguments.");
            auto* it = new MapIter;
            Value out(it);
            it->fn = a[0];
            for (int i = 1; i < n; ++i) it->iters.push_back(get_iter(a[i]));
            return out;
        }
        case T_filter: {
            no_keywords();
            if (n != 2) raise(T_TypeError, "filter expected 2 arguments, got " + std::to_string(n));
            auto* it = new FilterIter;
            Value out(it);
            it->fn = a[0];
            it->it = get_iter(a[1]);
            return out;
        }
        case T_zip: {
            only_keywords(c, {"strict"}, "zip");
            auto* it = new ZipIter;
            Value out(it);
            Value* strict = c.keyword("strict");
            it->strict = strict && truth(*strict);
            for (int i = 0; i < n; ++i) it->iters.push_back(get_iter(a[i]));
            return out;
        }
        case T_enumerate: {
            only_keywords(c, {"start", "iterable"}, "enumerate");
            auto* it = new EnumerateIter;
            Value out(it);
            Value* start = c.keyword("start");
            Value* iterable = c.keyword("iterable");
            if (!n && !iterable) raise(T_TypeError, "enumerate() missing required argument 'iterable'");
            it->it = get_iter(iterable ? *iterable : a[0]);
            it->n = start ? *start : n > 1 ? a[1] : Value::integer(0);
            if (!is_int(it->n)) raise(T_TypeError, quoted(it->n) + " object cannot be interpreted as an integer");
            if (it->n.tag == Tag::Bool) it->n = Value::integer(small_of(it->n));
            return out;
        }
        case T_reversed: {
            no_keywords();
            if (n != 1) raise(T_TypeError, "reversed expected 1 argument, got " + std::to_string(n));
            return b_reversed_of(a[0]);
        }
        case T_type: {
            if (n != 1) refuse("type() with three arguments");
            if (a[0].tag == Tag::Instance) return a[0].as<InstanceObj>()->cls;
            return Value::type(type_of(a[0]));
        }
        case T_deque: {
            only_keywords(c, {"iterable", "maxlen"}, "deque");
            auto* q = new DequeObj;
            Value out(q);
            Value* maxlen = c.keyword("maxlen");
            const Value limit = maxlen ? *maxlen : n > 1 ? a[1] : Value::none();
            if (!limit.is_none()) {
                if (!is_int(limit)) raise(T_TypeError, "an integer is required");
                q->maxlen = int_arg(limit);
                if (q->maxlen < 0) raise(T_ValueError, "maxlen must be non-negative");
            }
            Value* iterable = c.keyword("iterable");
            if (n || iterable) each(iterable ? *iterable : a[0], [&](Value& x) { deque_append(q, std::move(x)); });
            return out;
        }
        case T_Fraction: return make_fraction(a, n, kw);
        case T_chain: {
            no_keywords();
            auto* it = new ChainIter;
            Value out(it);
            Vals sources;
            for (int i = 0; i < n; ++i) sources.push_back(a[i]);
            it->outer = get_iter(make_list(std::move(sources)));
            return out;
        }
        case T_count: {
            only_keywords(c, {"start", "step"}, "count");
            auto* it = new CountIter;
            Value out(it);
            Value* start = c.keyword("start");
            Value* step = c.keyword("step");
            it->at = start ? *start : n > 0 ? a[0] : Value::integer(0);
            it->step = step ? *step : n > 1 ? a[1] : Value::integer(1);
            if (!is_number(it->at) && it->at.tag != Tag::Fraction) raise(T_TypeError, "a number is required");
            if (!is_number(it->step) && it->step.tag != Tag::Fraction) raise(T_TypeError, "a number is required");
            return out;
        }
        case T_islice: {
            no_keywords();
            if (n < 2 || n > 4) raise(T_TypeError, "islice expected at most 4 arguments, got " + std::to_string(n));
            auto* it = new IsliceIter;
            Value out(it);
            it->it = get_iter(a[0]);
            const char* const indices = "Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize.";
            const char* const stop_text = "Stop argument for islice() must be None or an integer: 0 <= x <= sys.maxsize.";
            const auto index = [&](const Value& v, int64_t none_value, const char* message) {
                if (v.is_none()) return none_value;
                if (!is_int(v) || !is_small(v)) raise(T_ValueError, message);
                if (small_of(v) < 0) raise(T_ValueError, small_of(v) == -1 ? message : indices);
                return small_of(v);
            };
            if (n == 2) {
                it->stop = index(a[1], -1, stop_text);
            } else {
                it->next_index = index(a[1], 0, indices);
                it->stop = index(a[2], -1, stop_text);
                if (n == 4) {
                    it->step = index(a[3], 1, "Step for islice() must be a positive integer or None.");
                    if (it->step < 1) raise(T_ValueError, "Step for islice() must be a positive integer or None.");
                }
            }
            return out;
        }
        case T_accumulate: {
            if (c.keyword("iterable")) refuse("accumulate(iterable=...)");
            only_keywords(c, {"func", "initial"}, "accumulate");
            auto* it = new AccumulateIter;
            Value out(it);
            Value* fn = c.keyword("func");
            Value* initial = c.keyword("initial");
            if (n < 1 || n > 2) raise(T_TypeError, "accumulate() takes 1 or 2 positional arguments");
            it->it = get_iter(a[0]);
            it->fn = fn ? *fn : n > 1 ? a[1] : Value::none();
            if (initial && !initial->is_none()) it->initial = *initial;
            return out;
        }
        case T_permutations: {
            if (nkw) refuse("keyword arguments to permutations");
            if (n < 1 || n > 2) raise(T_TypeError, "permutations() takes 1 or 2 arguments");
            auto* it = new PermutationsIter;
            Value out(it);
            it->pool = to_vals(a[0]);
            const int64_t size = (int64_t) it->pool.size();
            if (n > 1 && !a[1].is_none() && !is_int(a[1])) raise(T_TypeError, "Expected int as r");
            it->r = n > 1 && !a[1].is_none() ? int_arg(a[1]) : size;
            if (it->r < 0) raise(T_ValueError, "r must be non-negative");
            if (it->r <= size) {
                it->indices = iota(size);
                it->cycles.resize((std::size_t) it->r);
                for (int64_t i = 0; i < it->r; ++i) it->cycles[(std::size_t) i] = size - i;
            }
            return out;
        }
        case T_combinations: case T_combinations_with_replacement: {
            if (nkw) refuse("keyword arguments to combinations");
            if (n != 2) raise(T_TypeError, std::string(kTypes[type].name) + "() takes exactly 2 arguments");
            auto* it = new CombinationsIter(type == T_combinations_with_replacement);
            Value out(it);
            it->pool = to_vals(a[0]);
            it->r = int_arg(a[1]);
            if (it->r < 0) raise(T_ValueError, "r must be non-negative");
            return out;
        }
        case T_product: {
            only_keywords(c, {"repeat"}, "product");
            auto* it = new ProductIter;
            Value out(it);
            Value* repeat_kw = c.keyword("repeat");
            const int64_t repeat = repeat_kw ? int_arg(*repeat_kw) : 1;
            if (repeat < 0) raise(T_ValueError, "repeat argument cannot be negative");
            std::vector<Vals, Alloc<Vals>> first;
            for (int i = 0; i < n; ++i) first.push_back(to_vals(a[i]));
            if ((double) repeat * (double) (first.size() + 1) * sizeof(Vals) > (double) rt->limits.heap_bytes) too_big();
            for (int64_t r = 0; r < repeat; ++r)
                for (const auto& pool : first) {
                    rt->tick(pool.size() >> 3);
                    it->pools.push_back(pool);
                }
            return out;
        }
        case T_object: refuse("object()");
        default: refuse(std::string("calling ") + kTypes[type].name);
    }
}
inline Value chain_from_iterable(Call& c) {
    auto* it = new ChainIter;
    Value out(it);
    it->outer = get_iter(c.a[0]);
    return out;
}

inline double math_arg(const Value& v) {
    if (is_number(v) || v.tag == Tag::Fraction) return to_float(v);
    raise(T_TypeError, "must be real number, not " + std::string(type_name(v)));
}
using Unary = double (*)(double);
inline Value math1(const Value& v, Unary f, bool can_overflow) {
    const double x = math_arg(v), r = f(x);
    if (std::isnan(r) && !std::isnan(x)) raise(T_ValueError, "math domain error");
    if (std::isinf(r) && std::isfinite(x)) {
        if (can_overflow) raise(T_OverflowError, "math range error");
        raise(T_ValueError, "math domain error");
    }
    return Value::real(r);
}
#define SANDBOX_MATH1(name, fn, over) \
    inline Value m_##name(Call& c) { return math1(c.a[0], [](double x) { return (double) fn(x); }, over); }
SANDBOX_MATH1(sqrt, std::sqrt, false)
SANDBOX_MATH1(exp, std::exp, true)
SANDBOX_MATH1(expm1, std::expm1, true)
SANDBOX_MATH1(log1p, std::log1p, false)
SANDBOX_MATH1(sin, std::sin, false)
SANDBOX_MATH1(cos, std::cos, false)
SANDBOX_MATH1(tan, std::tan, false)
SANDBOX_MATH1(asin, std::asin, false)
SANDBOX_MATH1(acos, std::acos, false)
SANDBOX_MATH1(atan, std::atan, false)
SANDBOX_MATH1(sinh, std::sinh, true)
SANDBOX_MATH1(cosh, std::cosh, true)
SANDBOX_MATH1(tanh, std::tanh, false)
SANDBOX_MATH1(asinh, std::asinh, false)
SANDBOX_MATH1(acosh, std::acosh, false)
SANDBOX_MATH1(atanh, std::atanh, false)
SANDBOX_MATH1(fabs, std::fabs, false)
#undef SANDBOX_MATH1
inline Value m_floor(Call& c) {
    const Value& v = c.a[0];
    if (is_int(v)) return v.tag == Tag::Bool ? Value::integer(small_of(v)) : v;
    if (v.tag == Tag::Fraction) return int_floordiv(v.as<FractionObj>()->num, v.as<FractionObj>()->den);
    const double x = math_arg(v);
    if (std::isnan(x)) raise(T_ValueError, "cannot convert float NaN to integer");
    if (std::isinf(x)) raise(T_OverflowError, "cannot convert float infinity to integer");
    return int_from_double(std::floor(x));
}
inline Value m_ceil(Call& c) {
    const Value& v = c.a[0];
    if (is_int(v)) return v.tag == Tag::Bool ? Value::integer(small_of(v)) : v;
    if (v.tag == Tag::Fraction) return int_negate(int_floordiv(int_negate(v.as<FractionObj>()->num), v.as<FractionObj>()->den));
    const double x = math_arg(v);
    if (std::isnan(x)) raise(T_ValueError, "cannot convert float NaN to integer");
    if (std::isinf(x)) raise(T_OverflowError, "cannot convert float infinity to integer");
    return int_from_double(std::ceil(x));
}
inline Value m_trunc(Call& c) {
    if (is_int(c.a[0]) || c.a[0].tag == Tag::Float || c.a[0].tag == Tag::Fraction) return to_int(c.a, 1, nullptr);
    raise(T_TypeError, std::string("type ") + type_name(c.a[0]) + " doesn't define __trunc__ method");
}
inline Value exact_int(const Value& v) {
    if (!is_int(v)) raise(T_TypeError, quoted(v) + " object cannot be interpreted as an integer");
    return v.tag == Tag::Bool ? Value::integer(small_of(v)) : v;
}
inline Value m_gcd(Call& c) {
    Value out = Value::integer(0);
    for (int i = 0; i < c.n; ++i) out = int_gcd(out, exact_int(c.a[i]));
    return out;
}
inline Value m_lcm(Call& c) {
    Value out = Value::integer(1);
    for (int i = 0; i < c.n; ++i) {
        const Value x = int_abs(exact_int(c.a[i]));
        if (int_is_zero(x) || int_is_zero(out)) {
            out = Value::integer(0);
            continue;
        }
        out = int_mul(int_floordiv(out, int_gcd(out, x)), x);
    }
    return out;
}
inline Value m_factorial(Call& c) {
    const Value n = exact_int(c.a[0]);
    if (int_is_neg(n)) raise(T_ValueError, "factorial() not defined for negative values");
    if (!is_small(n)) too_big();
    Value out = Value::integer(1);
    for (int64_t i = 2; i <= n.i; ++i) {
        rt->tick();
        out = int_mul(out, Value::integer(i));
    }
    return out;
}
inline Value m_comb(Call& c) {
    const Value n = exact_int(c.a[0]);
    Value k = exact_int(c.a[1]);
    if (int_is_neg(n)) raise(T_ValueError, "n must be a non-negative integer");
    if (int_is_neg(k)) raise(T_ValueError, "k must be a non-negative integer");
    if (int_cmp(k, n) > 0) return Value::integer(0);
    const Value other = int_sub(n, k);
    if (int_cmp(other, k) < 0) k = other;
    if (!is_small(k)) too_big();
    Value out = Value::integer(1);
    for (int64_t i = 1; i <= k.i; ++i) {
        rt->tick();
        out = int_floordiv(int_mul(out, int_add(int_sub(n, k), Value::integer(i))), Value::integer(i));
    }
    return out;
}
inline Value m_perm(Call& c) {
    const Value n = exact_int(c.a[0]);
    if (c.n < 2 || c.a[1].is_none()) return m_factorial(c);
    const Value k = exact_int(c.a[1]);
    if (int_is_neg(n)) raise(T_ValueError, "n must be a non-negative integer");
    if (int_is_neg(k)) raise(T_ValueError, "k must be a non-negative integer");
    if (int_cmp(k, n) > 0) return Value::integer(0);
    if (!is_small(k)) too_big();
    Value out = Value::integer(1);
    for (int64_t i = 0; i < k.i; ++i) {
        rt->tick();
        out = int_mul(out, int_sub(n, Value::integer(i)));
    }
    return out;
}
inline Value m_isqrt(Call& c) {
    const Value n = exact_int(c.a[0]);
    if (int_is_neg(n)) raise(T_ValueError, "isqrt() argument must be nonnegative");
    if (int_is_zero(n)) return n;
    Value x = int_shl(Value::integer(1), (int_bits(n) + 1) / 2);
    for (;;) {
        rt->tick();
        Value y = int_shr(int_add(x, int_floordiv(n, x)), 1);
        if (int_cmp(y, x) >= 0) return x;
        x = std::move(y);
    }
}
inline Value m_pow(Call& c) {
    const double x = math_arg(c.a[0]), y = math_arg(c.a[1]);
    double r = 0;
    bool domain = false, range = false;
    if (!std::isfinite(x) || !std::isfinite(y)) {
        if (std::isnan(x)) r = y == 0 ? 1.0 : x;
        else if (std::isnan(y)) r = x == 1 ? 1.0 : y;
        else if (std::isinf(x)) {
            const bool odd = std::isfinite(y) && std::fmod(std::fabs(y), 2.0) == 1.0;
            if (y > 0) r = odd ? x : std::fabs(x);
            else if (y == 0) r = 1.0;
            else r = odd ? std::copysign(0.0, x) : 0.0;
        } else {
            if (std::fabs(x) == 1.0) r = 1.0;
            else if (y > 0 && std::fabs(x) > 1.0) r = y;
            else if (y < 0 && std::fabs(x) < 1.0) {
                r = -y;
                domain = x == 0;
            } else r = 0.0;
        }
    } else {
        r = std::pow(x, y);
        if (std::isnan(r)) domain = true;
        else if (std::isinf(r)) {
            if (x == 0) domain = true;
            else range = true;
        }
    }
    if (domain) raise(T_ValueError, "math domain error");
    if (range) raise(T_OverflowError, "math range error");
    return Value::real(r);
}
using LogFn = double (*)(double);
inline double log_of(const Value& v, LogFn f) {
    if (is_int(v)) {
        if (int_is_neg(v) || int_is_zero(v)) raise(T_ValueError, "math domain error");
        if (int_bits(v) <= 1000) return f(int_to_double(v));
        const std::size_t bits = int_bits(v);
        const Value top = int_shr(v, bits - 62);
        return f(std::ldexp((double) (uint64_t) top.i, -62)) + f(2.0) * (double) bits;
    }
    const double x = math_arg(v);
    if (std::isnan(x)) return x;
    if (std::isinf(x)) {
        if (x > 0) return x;
        raise(T_ValueError, "math domain error");
    }
    if (x <= 0) raise(T_ValueError, "math domain error");
    return f(x);
}
inline Value m_log(Call& c) {
    const double num = log_of(c.a[0], [](double x) { return (double) std::log(x); });
    if (c.n < 2) return Value::real(num);
    const double den = log_of(c.a[1], [](double x) { return (double) std::log(x); });
    if (den == 0) raise(T_ZeroDivisionError, "float division by zero");
    return Value::real(num / den);
}
// CPython's own m_log2 where the platform has no log2, as on Windows. libm's log2 differs from it in the last bit.
inline double log2_of(double x) {
    int e = 0;
    const double m = std::frexp(x, &e);
    if (x >= 1.0) return std::log(2.0 * m) / std::log(2.0) + (double) (e - 1);
    return std::log(m) / std::log(2.0) + (double) e;
}
inline Value m_log2(Call& c) { return Value::real(log_of(c.a[0], log2_of)); }
inline Value m_log10(Call& c) { return Value::real(log_of(c.a[0], [](double x) { return (double) std::log10(x); })); }
inline Value m_atan2(Call& c) { return Value::real(std::atan2(math_arg(c.a[0]), math_arg(c.a[1]))); }
inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline Value m_degrees(Call& c) { return Value::real(math_arg(c.a[0]) * (180.0 / kPi)); }
inline Value m_radians(Call& c) { return Value::real(math_arg(c.a[0]) * (kPi / 180.0)); }
inline Value m_isclose(Call& c) {
    only_keywords(c, {"a", "b", "rel_tol", "abs_tol"}, "isclose");
    Value* rel_kw = c.keyword("rel_tol");
    Value* abs_kw = c.keyword("abs_tol");
    Value* a_kw = c.keyword("a");
    Value* b_kw = c.keyword("b");
    if (c.n + (a_kw ? 1 : 0) + (b_kw ? 1 : 0) != 2) raise(T_TypeError, "isclose() missing required argument");
    const double a = math_arg(a_kw ? *a_kw : c.a[0]), b = math_arg(b_kw ? *b_kw : c.n > 1 ? c.a[1] : c.a[0]);
    const double rel = rel_kw ? math_arg(*rel_kw) : 1e-9, tol = abs_kw ? math_arg(*abs_kw) : 0.0;
    if (rel < 0 || tol < 0) raise(T_ValueError, "tolerances must be non-negative");
    if (a == b) return Value::boolean(true);
    if (std::isinf(a) || std::isinf(b)) return Value::boolean(false);
    const double diff = std::fabs(b - a);
    return Value::boolean(diff <= std::fabs(rel * b) || diff <= std::fabs(rel * a) || diff <= tol);
}
inline Value m_isnan(Call& c) { return Value::boolean(std::isnan(math_arg(c.a[0]))); }
inline Value m_isinf(Call& c) { return Value::boolean(std::isinf(math_arg(c.a[0]))); }
inline Value m_isfinite(Call& c) { return Value::boolean(std::isfinite(math_arg(c.a[0]))); }
inline Value m_copysign(Call& c) { return Value::real(std::copysign(math_arg(c.a[0]), math_arg(c.a[1]))); }
inline Value m_fmod(Call& c) {
    const double x = math_arg(c.a[0]), y = math_arg(c.a[1]);
    if (std::isinf(y) && std::isfinite(x)) return Value::real(x);
    const double r = std::fmod(x, y);
    if (std::isnan(r) && !std::isnan(x) && !std::isnan(y)) raise(T_ValueError, "math domain error");
    return Value::real(r);
}
inline Value m_modf(Call& c) {
    const double x = math_arg(c.a[0]);
    if (std::isinf(x)) return pair(Value::real(std::copysign(0.0, x)), Value::real(x));
    if (std::isnan(x)) return pair(Value::real(x), Value::real(x));
    double whole = 0;
    const double part = std::modf(x, &whole);
    return pair(Value::real(part), Value::real(whole));
}
inline Value m_fsum(Call& c) {
    std::vector<double, Alloc<double>> p;
    double special = 0, inf_sum = 0;
    each(c.a[0], [&](Value& item) {
        double x = math_arg(item);
        const double original = x;
        std::size_t i = 0;
        for (std::size_t j = 0; j < p.size(); ++j) {
            double y = p[j];
            if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
            const double hi = x + y, yr = hi - x, lo = y - yr;
            if (lo != 0) p[i++] = lo;
            x = hi;
        }
        rt->tick(p.size() >> 4);
        p.resize(i);
        if (x != 0) {
            if (!std::isfinite(x)) {
                if (std::isfinite(original)) raise(T_OverflowError, "intermediate overflow in fsum");
                if (std::isinf(original)) inf_sum += original;
                special += original;
                p.clear();
            } else {
                p.push_back(x);
            }
        }
    });
    if (special != 0) {
        if (std::isnan(inf_sum)) raise(T_ValueError, "-inf + inf in fsum");
        return Value::real(special);
    }
    double hi = 0, lo = 0;
    std::size_t n = p.size();
    if (n > 0) {
        hi = p[--n];
        while (n > 0) {
            const double x = hi, y = p[--n];
            hi = x + y;
            const double yr = hi - x;
            lo = y - yr;
            if (lo != 0) break;
        }
        if (n > 0 && ((lo < 0 && p[n - 1] < 0) || (lo > 0 && p[n - 1] > 0))) {
            const double y = lo * 2.0, x = hi + y, yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return Value::real(hi);
}
// CPython's vector_norm, with its compensated summation: libm's hypot can differ in the last bit.
inline double vector_norm(const std::vector<double, Alloc<double>>& v, double max_value, bool found_nan) {
    constexpr double T27 = 134217729.0;
    if (std::isinf(max_value)) return max_value;
    if (found_nan) return std::numeric_limits<double>::quiet_NaN();
    if (max_value == 0 || v.size() <= 1) return max_value;
    int max_e = 0;
    std::frexp(max_value, &max_e);
    if (max_e < -1023) {
        double sum = 1.0, frac = 0;
        for (double x : v) {
            x /= max_value;
            x *= x;
            const double old = sum;
            sum += x;
            frac += (old - sum) + x;
        }
        return max_value * std::sqrt(sum - 1.0 + frac);
    }
    const double scale = std::ldexp(1.0, -max_e);
    double csum = 1.0, frac1 = 0, frac2 = 0, frac3 = 0;
    for (double x : v) {
        x *= scale;
        double t = x * T27;
        const double hi = t - (t - x), lo = x - hi;
        x = hi * hi;
        double old = csum;
        csum += x;
        frac1 += (old - csum) + x;
        x = 2.0 * hi * lo;
        old = csum;
        csum += x;
        frac2 += (old - csum) + x;
        frac3 += lo * lo;
    }
    const double h = std::sqrt(csum - 1.0 + (frac1 + frac2 + frac3));
    double x = h;
    const double t = x * T27;
    const double hi = t - (t - x), lo = x - hi;
    x = -hi * hi;
    double old = csum;
    csum += x;
    frac1 += (old - csum) + x;
    x = -2.0 * hi * lo;
    old = csum;
    csum += x;
    frac2 += (old - csum) + x;
    x = -lo * lo;
    old = csum;
    csum += x;
    frac3 += (old - csum) + x;
    x = csum - 1.0 + (frac1 + frac2 + frac3);
    return (h + x / (2.0 * h)) / scale;
}
inline Value m_hypot(Call& c) {
    std::vector<double, Alloc<double>> v;
    double top = 0;
    bool found_nan = false;
    for (int i = 0; i < c.n; ++i) {
        const double x = std::fabs(math_arg(c.a[i]));
        found_nan = found_nan || std::isnan(x);
        top = std::max(top, x);
        v.push_back(x);
    }
    return Value::real(vector_norm(v, top, found_nan));
}
inline Value m_prod(Call& c) {
    only_keywords(c, {"start"}, "prod");
    Value* start = c.keyword("start");
    Value out = start ? *start : Value::integer(1);
    each(c.a[0], [&](Value& x) { out = binary(Bin::Mul, out, x, false); });
    return out;
}
inline Value f_reduce(Call& c) {
    Value it = get_iter(c.a[1]);
    Value total = c.n > 2 ? c.a[2] : Value();
    Value x;
    while (iter_next(it, x)) {
        rt->tick();
        if (total.empty()) {
            total = std::move(x);
            continue;
        }
        Value args[2] = {std::move(total), std::move(x)};
        total = call_value(c.a[0], args, 2);
    }
    if (total.empty()) raise(T_TypeError, "reduce() of empty iterable with no initial value");
    return total;
}
inline Value f_wrap(Call& c);
inline const NativeDef kWrapWith = {"lru_cache", f_wrap, 2, 2};
inline Value cached_of(const Value& fn, int64_t maxsize) {
    if (fn.tag != Tag::Function && fn.tag != Tag::Native && fn.tag != Tag::Type && fn.tag != Tag::Bound && fn.tag != Tag::Cached && fn.tag != Tag::Class && fn.tag != Tag::Method) raise(T_TypeError, "the first argument must be callable");
    return Value(new CachedObj(fn, maxsize));
}
inline Value f_wrap(Call& c) { return cached_of(c.a[1], c.a[0].is_none() ? -1 : std::max<int64_t>(int_arg(c.a[0]), 0)); }
inline Value f_lru_cache(Call& c) {
    only_keywords(c, {"maxsize", "typed"}, "lru_cache");
    Value* typed = c.keyword("typed");
    if (typed && truth(*typed)) refuse("lru_cache(typed=True)");
    if (c.n > 1 && truth(c.a[1])) refuse("lru_cache(typed=True)");
    Value* size = c.keyword("maxsize");
    const Value first = size ? *size : c.n ? c.a[0] : Value::integer(128);
    if (!first.is_none() && !is_int(first)) {
        if (first.tag == Tag::Function || first.tag == Tag::Native || first.tag == Tag::Type || first.tag == Tag::Bound || first.tag == Tag::Method || first.tag == Tag::Class) return cached_of(first, 128);
        raise(T_TypeError, "Expected first argument to be an integer, a callable, or None");
    }
    return Value(new BoundObj(first, &kWrapWith));
}
inline Value f_cache(Call& c) { return cached_of(c.a[0], -1); }
inline Value t_cast(Call& c) { return c.a[1]; }

// The sifts of CPython's heapq.py, so ties land where they land there.
inline Vals& heap_arg(const Value& v) {
    if (v.tag != Tag::List) raise(T_TypeError, "heap argument must be a list");
    return v.as<ListObj>()->v;
}
inline void heap_down(Vals& h, std::size_t start, std::size_t pos) {
    Value item = std::move(h[pos]);
    while (pos > start) {
        const std::size_t up = (pos - 1) >> 1;
        rt->tick();
        if (!order(Cmp::Lt, item, h[up])) break;
        h[pos] = h[up];
        pos = up;
    }
    h[pos] = std::move(item);
}
inline void heap_up(Vals& h, std::size_t pos) {
    const std::size_t end = h.size(), start = pos;
    Value item = std::move(h[pos]);
    for (std::size_t child = 2 * pos + 1; child < end; child = 2 * pos + 1) {
        rt->tick();
        if (child + 1 < end && !order(Cmp::Lt, h[child], h[child + 1])) ++child;
        h[pos] = h[child];
        pos = child;
    }
    h[pos] = std::move(item);
    heap_down(h, start, pos);
}
inline Value hq_push(Call& c) {
    Vals& h = heap_arg(c.a[0]);
    h.push_back(c.a[1]);
    heap_down(h, 0, h.size() - 1);
    return Value::none();
}
inline Value hq_pop(Call& c) {
    Vals& h = heap_arg(c.a[0]);
    if (h.empty()) raise(T_IndexError, "index out of range");
    Value last = std::move(h.back());
    h.pop_back();
    if (h.empty()) return last;
    Value top = std::move(h[0]);
    h[0] = std::move(last);
    heap_up(h, 0);
    return top;
}
inline Value hq_heapify(Call& c) {
    Vals& h = heap_arg(c.a[0]);
    for (std::size_t i = h.size() / 2; i-- > 0;) heap_up(h, i);
    return Value::none();
}
inline Value hq_replace(Call& c) {
    Vals& h = heap_arg(c.a[0]);
    if (h.empty()) raise(T_IndexError, "index out of range");
    Value top = std::move(h[0]);
    h[0] = c.a[1];
    heap_up(h, 0);
    return top;
}
inline Value hq_pushpop(Call& c) {
    Vals& h = heap_arg(c.a[0]);
    if (h.empty() || !order(Cmp::Lt, h[0], c.a[1])) return c.a[1];
    Value top = std::move(h[0]);
    h[0] = c.a[1];
    heap_up(h, 0);
    return top;
}
inline Value heap_first(Call& c, const char* name, bool reverse) {
    only_keywords(c, {"key"}, name);
    Vals items = to_vals(c.a[1]);
    Value* key = c.keyword("key");
    sort_values(items, key ? *key : Value::none(), reverse);
    const int64_t n = std::max<int64_t>(int_arg(c.a[0]), 0);
    if ((int64_t) items.size() > n) items.resize((std::size_t) n);
    return make_list(std::move(items));
}
inline Value hq_nlargest(Call& c) { return heap_first(c, "nlargest", true); }
inline Value hq_nsmallest(Call& c) { return heap_first(c, "nsmallest", false); }

inline int64_t bisect_at(Call& c, const char* name, bool right) {
    only_keywords(c, {"lo", "hi", "key"}, name);
    if (c.a[0].tag != Tag::List && c.a[0].tag != Tag::Tuple) refuse(std::string(name) + " on this sequence");
    const Vals& a = seq_of(c.a[0]);
    Value* lo_kw = c.keyword("lo");
    Value* hi_kw = c.keyword("hi");
    Value* key_kw = c.keyword("key");
    const Value* lo_v = c.n > 2 ? &c.a[2] : lo_kw;
    const Value* hi_v = c.n > 3 ? &c.a[3] : hi_kw;
    int64_t lo = lo_v ? int_arg(*lo_v) : 0;
    if (lo < 0) raise(T_ValueError, "lo must be non-negative");
    int64_t hi = hi_v && !hi_v->is_none() ? int_arg(*hi_v) : (int64_t) a.size();
    const Value key = key_kw ? *key_kw : Value::none();
    const bool insort = std::string_view(name).starts_with("insort");
    Value x = c.a[1];
    if (insort && !key.is_none()) {
        Value arg = x;
        x = call_value(key, &arg, 1);
    }
    while (lo < hi) {
        const int64_t mid = (lo + hi) / 2;
        rt->tick();
        if (mid >= (int64_t) a.size()) raise(T_IndexError, "list index out of range");
        Value at = a[(std::size_t) mid];
        if (!key.is_none()) at = call_value(key, &at, 1);
        if (right ? order(Cmp::Lt, x, at) : !order(Cmp::Lt, at, x)) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}
inline Value s_left(Call& c) { return Value::integer(bisect_at(c, "bisect_left", false)); }
inline Value s_right(Call& c) { return Value::integer(bisect_at(c, "bisect_right", true)); }
inline Value insort_at(Call& c, const char* name, bool right) {
    if (c.a[0].tag != Tag::List) raise(T_AttributeError, "object has no attribute 'insert'");
    const int64_t at = bisect_at(c, name, right);
    Vals& a = c.a[0].as<ListObj>()->v;
    rt->tick((uint64_t) ((int64_t) a.size() - at) >> 4);
    a.insert(a.begin() + at, c.a[1]);
    return Value::none();
}
inline Value s_insort_left(Call& c) { return insort_at(c, "insort_left", false); }
inline Value s_insort_right(Call& c) { return insort_at(c, "insort_right", true); }

inline const NativeDef kBuiltins[] = {
    {"print", b_print, 0, -1, true}, {"len", b_len, 1, 1}, {"abs", b_abs, 1, 1}, {"sum", b_sum, 1, 2, true}, {"min", b_min, 1, -1, true},
    {"max", b_max, 1, -1, true}, {"sorted", b_sorted, 1, 1, true}, {"any", b_any, 1, 1}, {"all", b_all, 1, 1}, {"repr", b_repr, 1, 1},
    {"ascii", b_ascii, 1, 1}, {"round", b_round, 0, 2, true}, {"pow", b_pow, 0, 3, true}, {"divmod", b_divmod, 2, 2}, {"ord", b_ord, 1, 1},
    {"chr", b_chr, 1, 1}, {"hex", b_hex, 1, 1}, {"oct", b_oct, 1, 1}, {"bin", b_bin, 1, 1}, {"isinstance", b_isinstance, 2, 2},
    {"issubclass", b_issubclass, 2, 2}, {"callable", b_callable, 1, 1}, {"hash", b_hash, 1, 1}, {"iter", b_iter, 1, 2}, {"next", b_next, 1, 2},
    {"format", b_format, 1, 2},
};
inline const NativeDef kMath[] = {
    {"sqrt", m_sqrt, 1, 1}, {"exp", m_exp, 1, 1}, {"expm1", m_expm1, 1, 1}, {"log1p", m_log1p, 1, 1}, {"sin", m_sin, 1, 1}, {"cos", m_cos, 1, 1},
    {"tan", m_tan, 1, 1}, {"asin", m_asin, 1, 1}, {"acos", m_acos, 1, 1}, {"atan", m_atan, 1, 1}, {"sinh", m_sinh, 1, 1}, {"cosh", m_cosh, 1, 1},
    {"tanh", m_tanh, 1, 1}, {"asinh", m_asinh, 1, 1}, {"acosh", m_acosh, 1, 1}, {"atanh", m_atanh, 1, 1}, {"fabs", m_fabs, 1, 1},
    {"floor", m_floor, 1, 1}, {"ceil", m_ceil, 1, 1}, {"trunc", m_trunc, 1, 1}, {"gcd", m_gcd, 0, -1}, {"lcm", m_lcm, 0, -1},
    {"factorial", m_factorial, 1, 1}, {"comb", m_comb, 2, 2}, {"perm", m_perm, 1, 2}, {"isqrt", m_isqrt, 1, 1}, {"pow", m_pow, 2, 2},
    {"log", m_log, 1, 2}, {"log2", m_log2, 1, 1}, {"log10", m_log10, 1, 1}, {"atan2", m_atan2, 2, 2}, {"degrees", m_degrees, 1, 1},
    {"radians", m_radians, 1, 1}, {"isclose", m_isclose, 0, 2, true}, {"isnan", m_isnan, 1, 1}, {"isinf", m_isinf, 1, 1},
    {"isfinite", m_isfinite, 1, 1}, {"copysign", m_copysign, 2, 2}, {"fmod", m_fmod, 2, 2}, {"modf", m_modf, 1, 1}, {"fsum", m_fsum, 1, 1},
    {"hypot", m_hypot, 0, -1}, {"prod", m_prod, 1, 1, true},
};
inline const NativeDef kChainFromIterable = {"from_iterable", chain_from_iterable, 1, 1, false, true};
inline const NativeDef* chain_from_iterable_def() { return &kChainFromIterable; }
inline const NativeDef kFunctools[] = {{"reduce", f_reduce, 2, 3}, {"lru_cache", f_lru_cache, 0, 2, true}, {"cache", f_cache, 1, 1}};
inline const NativeDef kHeapq[] = {{"heappush", hq_push, 2, 2}, {"heappop", hq_pop, 1, 1}, {"heapify", hq_heapify, 1, 1}, {"heapreplace", hq_replace, 2, 2},
    {"heappushpop", hq_pushpop, 2, 2}, {"nlargest", hq_nlargest, 2, 2, true}, {"nsmallest", hq_nsmallest, 2, 2, true}};
inline const NativeDef kBisect[] = {{"bisect_left", s_left, 2, 4, true}, {"bisect_right", s_right, 2, 4, true}, {"bisect", s_right, 2, 4, true},
    {"insort_left", s_insort_left, 2, 4, true}, {"insort_right", s_insort_right, 2, 4, true}, {"insort", s_insort_right, 2, 4, true}};
inline const NativeDef kCast = {"cast", t_cast, 2, 2};
inline constexpr std::string_view kTypingNames =
    " List Dict Tuple Set FrozenSet Optional Union Any Callable Iterable Iterator Sequence Mapping MutableMapping MutableSequence "
    "Generator Type TypeVar Generic NoReturn Final Literal DefaultDict Deque Counter Collection Hashable Sized ";
enum ModuleId : int { kModuleMath, kModuleItertools, kModuleFunctools, kModuleCollections, kModuleFractions, kModuleTyping, kModuleHeapq, kModuleBisect };
inline constexpr const char* kModuleNames[] = {"math", "itertools", "functools", "collections", "fractions", "typing", "heapq", "bisect"};
inline Value import_module(std::string_view name) {
    for (int i = 0; i < (int) std::size(kModuleNames); ++i)
        if (name == kModuleNames[i]) return Value::module(i);
    const std::string_view top = name.substr(0, name.find('.'));
    for (const char* known : kModuleNames)
        if (top == known) refuse(std::string(name));
    raise(T_ModuleNotFoundError, "No module named '" + std::string(top) + "'");
}
template <std::size_t N>
inline const NativeDef* find_native(const NativeDef (&table)[N], std::string_view name) {
    for (const auto& d : table)
        if (name == d.name) return &d;
    return nullptr;
}
inline Value module_attr(int module, std::string_view name) {
    const std::string whole = std::string(kModuleNames[module]) + "." + std::string(name);
    switch (module) {
        case kModuleMath: {
            if (name == "pi") return Value::real(kPi);
            if (name == "e") return Value::real(2.718281828459045235360287471352662498);
            if (name == "tau") return Value::real(2 * kPi);
            if (name == "inf") return Value::real(HUGE_VAL);
            if (name == "nan") return Value::real(std::numeric_limits<double>::quiet_NaN());
            if (const NativeDef* d = find_native(kMath, name)) return Value::from(d);
            break;
        }
        case kModuleItertools:
            if (name == "permutations") return Value::type(T_permutations);
            if (name == "combinations") return Value::type(T_combinations);
            if (name == "combinations_with_replacement") return Value::type(T_combinations_with_replacement);
            if (name == "product") return Value::type(T_product);
            if (name == "accumulate") return Value::type(T_accumulate);
            if (name == "chain") return Value::type(T_chain);
            if (name == "count") return Value::type(T_count);
            if (name == "islice") return Value::type(T_islice);
            break;
        case kModuleFunctools:
            if (const NativeDef* d = find_native(kFunctools, name)) return Value::from(d);
            break;
        case kModuleHeapq:
            if (const NativeDef* d = find_native(kHeapq, name)) return Value::from(d);
            break;
        case kModuleBisect:
            if (const NativeDef* d = find_native(kBisect, name)) return Value::from(d);
            break;
        case kModuleCollections:
            if (name == "Counter") return Value::type(T_Counter);
            if (name == "defaultdict") return Value::type(T_defaultdict);
            if (name == "deque") return Value::type(T_deque);
            break;
        case kModuleFractions:
            if (name == "Fraction") return Value::type(T_Fraction);
            break;
        default:
            if (name == "cast") return Value::from(&kCast);
            if (name == "TYPE_CHECKING") return Value::boolean(false);
            if (kTypingNames.find(" " + std::string(name) + " ") != std::string_view::npos) return Value::of(Tag::Stub);
    }
    refuse(whole);
}
inline Value builtin_value(std::string_view name) {
    if (name == "__name__") return text(std::string_view("__main__"));
    if (const NativeDef* d = find_native(kBuiltins, name)) return Value::from(d);
    // Types Python names only inside a module, or nowhere, are not builtins.
    for (int t = 0; t < kTypeCount; ++t) {
        if (name != kTypes[t].name) continue;
        if (t == T_function || t == T_builtin_function_or_method || t == T_method || t == T_module || t == T_NoneType || t == T_iterator || t == T_ellipsis || t == T_generator) return Value();
        if ((t >= T_dict_keys && t <= T_dict_items) || (t >= T_Counter && t <= T__lru_cache_wrapper) || (t >= T_permutations && t <= T_islice)) return Value();
        return Value::type(t);
    }
    return Value();
}
inline void teardown() {
    for (Obj* o = rt->all; o; o = o->next) ++o->rc;
    for (Obj* o = rt->all; o; o = o->next) o->clear();
    for (Obj* o = rt->all; o;) {
        Obj* next = o->next;
        if (--o->rc == 0) destroy(o);
        o = next;
    }
}
using SourceText = std::basic_string<char, std::char_traits<char>, Alloc<char>>;
inline std::string one_line(std::string text_value) {
    for (auto& c : text_value)
        if (c == '\n' || c == '\r') c = ' ';
    if (text_value.size() > 300) text_value.resize(300);
    return text_value;
}
inline void program_body(std::string_view source, Result& result) {
    SourceText text_buffer;
    text_buffer.reserve(source.size() + 2);
    std::size_t i = source.size() >= 3 && source.substr(0, 3) == "\xEF\xBB\xBF" ? 3 : 0;
    for (; i < source.size(); ++i) {
        const char c = source[i];
        if (c == '\0') throw Stop{Status::Syntax, "SyntaxError: source code cannot contain null bytes (line 1)"};
        if (c == '\r') {
            if (i + 1 < source.size() && source[i + 1] == '\n') ++i;
            text_buffer.push_back('\n');
        } else {
            text_buffer.push_back(c);
        }
    }
    if (text_buffer.empty() || text_buffer.back() != '\n') text_buffer.push_back('\n');
    const std::string_view text_view(text_buffer.data(), text_buffer.size());
    {
        U32 check;
        if (!decode_utf8(text_view, check)) throw Stop{Status::Syntax, "SyntaxError: source is not valid UTF-8 (line 1)"};
    }
    Lexer lexer(text_view, 0, text_view.size(), 1, 0);
    lexer.run();
    Program program;
    Parser parser(program, text_view, &lexer.out, &lexer.error);
    Node* tree = parser.parse_file();
    Compiler compiler(program);
    Code* module = compiler.compile_module(tree);
    VM machine(program, compiler);
    VM* const outer = vm;
    vm = &machine;
    struct Restore {
        VM* previous;
        ~Restore() { vm = previous; }
    } restore{outer};
    try {
        auto* fn = new FunctionObj(module);
        const Value func(fn);
        machine.enter(func, nullptr, 0, nullptr);
        machine.run(0);
    } catch (const PyError& e) {
        auto* exc = e.exc.as<ExcObj>();
        if (exc->type == T_SystemExit) return;
        result.status = Status::Raised;
        std::string message;
        try {
            message = utf8(str_of(str_value(e.exc)));
        } catch (const Stop&) {
            message = "";
        } catch (const PyError&) {
            message = "";
        }
        std::string line = kTypes[exc->type].name;
        if (!message.empty()) line += ": " + message;
        result.error = one_line(line) + " (line " + std::to_string(exc->line) + ")";
    } catch (const Stop& stop) {
        result.status = stop.status;
        result.error = one_line(stop.error) + " (line " + std::to_string(std::max(machine.current_line(), 1)) + ")";
    }
}
}  // namespace detail

inline Result run(std::string_view source, const Limits& limits, const std::atomic<bool>* cancel) {
    using namespace detail;
    Result result;
    Runtime runtime;
    runtime.limits = limits;
    runtime.cancel = cancel;
    runtime.next_poll = std::min<uint64_t>(1024, limits.steps + 1);
    const char stack_mark = 0;
    runtime.stack_base = (std::uintptr_t) &stack_mark;
    Runtime* const outer = rt;
    rt = &runtime;
    try {
        program_body(source, result);
    } catch (const Stop& stop) {
        result.status = stop.status;
        result.error = stop.error;
    } catch (const std::bad_alloc&) {
        result.status = Status::MemoryLimit;
        result.error = "MemoryError: the allocator refused";
    } catch (const std::length_error&) {
        result.status = Status::MemoryLimit;
        result.error = "MemoryError: a size past what a container holds";
    } catch (const std::exception& e) {
        result.status = Status::Unsupported;
        result.error = std::string("unsupported: an internal error: ") + e.what();
    } catch (const PyError&) {
        result.status = Status::Unsupported;
        result.error = "unsupported: an exception outside the program";
    }
    teardown();
    result.output = std::move(runtime.output);
    result.truncated = runtime.truncated;
    result.steps = runtime.steps;
    result.peak_bytes = runtime.peak;
    rt = outer;
    return result;
}
}  // namespace strata::igc::sandbox
#undef SANDBOX_TYPES
