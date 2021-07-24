#include <vector>

// bpo-39573: Make Py_IS_TYPE take const args. Add _PyObject_CAST_CONST. #18799
#if PY_VERSION_HEX < 0x03090000
#   define _PyObject_CAST_CONST(op) ((const PyObject*)(op))

    static inline int _Py_IS_TYPE(const PyObject *ob, const PyTypeObject *type) {
        return ob->ob_type == type;
    }

#   define Py_IS_TYPE(ob, type) _Py_IS_TYPE(_PyObject_CAST_CONST(ob), type)
#endif

typedef std::vector<PyObject *> objectstack;

int _validate_dict(PyObject *main, PyObject *rest, objectstack *stack=NULL);
int _validate_tuple(PyObject *main, PyObject *rest, objectstack *stack=NULL);
int _validate_list(PyObject *main, PyObject *rest, objectstack *stack=NULL);

PyObject* validate(PyObject *self, PyObject *args);

extern const PyMethodDef def_validate;
