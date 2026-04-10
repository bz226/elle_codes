typedef struct {
    int num_ints;
    int num_toggles;
    int num_floats;
    Widget p_u_shell;
} OptionDataItem;

typedef struct {
    int id;
    int type;
} OptionItem;

typedef struct {
    int id;
} ToggleItem;

typedef struct {
    char *label;
    void (*callback)();
    OptionDataItem *data;
} OptsActionAreaItem;
