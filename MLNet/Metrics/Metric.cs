using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     Metric
    /// </summary>
    public abstract class Metric
    {
        public double Value { protected set; get; }

        public double Call(NDarray y_true, NDarray y_pred)
        {
            Value = call(y_true, y_pred);
            return Value;
        }

        internal abstract double call(NDarray y_true, NDarray y_pred);

        public override string ToString()
        {
            return $"{GetType().Name}:\t{Value:F4}";
        }
    }
}