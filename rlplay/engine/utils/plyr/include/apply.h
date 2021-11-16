int parse_apply_args(PyObject *args, PyObject **callable, PyObject **main, PyObject **rest);

PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest,
                 bool const safe, bool const star, PyObject *kwargs,
                 PyObject *finalizer);

PyObject* apply(PyObject *self, PyObject *args, PyObject *kwargs);

extern const PyMethodDef def_apply;
