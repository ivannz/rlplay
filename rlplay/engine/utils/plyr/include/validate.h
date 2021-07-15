
int _validate_dict(PyObject *main, PyObject *rest, objectstack *stack=NULL);
int _validate_tuple(PyObject *main, PyObject *rest, objectstack *stack=NULL);
int _validate_list(PyObject *main, PyObject *rest, objectstack *stack=NULL);

static PyObject* validate(PyObject *self, PyObject *args);
