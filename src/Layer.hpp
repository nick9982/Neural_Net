class Layer
{
    public:
        virtual ~Layer() {}
        double (*act_function)(double);
        double (*act_function_derivative)(double);
};
