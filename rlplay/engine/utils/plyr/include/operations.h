PyObject* getitem(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* setitem(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* is_sequence(PyObject *self, PyObject *object);
PyObject* is_mapping(PyObject *self, PyObject *object);

extern const PyMethodDef def_getitem;
extern const PyMethodDef def_setitem;
extern const PyMethodDef def_is_sequence;
extern const PyMethodDef def_is_mapping;
