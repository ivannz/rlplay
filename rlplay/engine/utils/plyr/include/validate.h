#include <vector>

// bpo-39573: Make Py_IS_TYPE take const args. Add _PyObject_CAST_CONST. #18799
#if PY_VERSION_HEX < 0x03090000
#   define _PyObject_CAST_CONST(op) ((const PyObject*)(op))

    static inline int _Py_IS_TYPE(const PyObject *ob, const PyTypeObject *type) {
        return ob->ob_type == type;
    }

#   define Py_IS_TYPE(ob, type) _Py_IS_TYPE(_PyObject_CAST_CONST(ob), type)
#endif

template <class T>
struct Pyallocator
{
  typedef T value_type;

  Pyallocator () = default;
  template <class U> constexpr Pyallocator (const Pyallocator <U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();

    if (auto p = static_cast<T*>(PyMem_RawMalloc(n * sizeof(T)))) {
      return p;
    }

    throw std::bad_alloc();
  }

  void deallocate(T* p, std::size_t n) noexcept {
    PyMem_RawFree(p);
  }
};

template <class T, class U>
bool operator==(const Pyallocator <T>&, const Pyallocator <U>&) { return true; }
template <class T, class U>
bool operator!=(const Pyallocator <T>&, const Pyallocator <U>&) { return false; }

typedef std::vector<PyObject *, Pyallocator<PyObject *>> objectstack;

int _validate_dict(PyObject *main, PyObject *rest, objectstack *stack=NULL);
int _validate_tuple(PyObject *main, PyObject *rest, objectstack *stack=NULL);
int _validate_list(PyObject *main, PyObject *rest, objectstack *stack=NULL);

PyObject* validate(PyObject *self, PyObject *args);

extern const PyMethodDef def_validate;
