static const char *__doc__;

int parse_apply_args(PyObject *args, PyObject **callable, PyObject **main, PyObject **rest);

static PyObject* _apply(PyObject *callable, PyObject *main, PyObject *rest,
                        bool const safe, bool const star, PyObject *kwargs);