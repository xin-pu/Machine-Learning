using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     Metric
    /// </summary>
    public abstract class Metric
    {
        public double metric { protected set; get; }

        public double Call(NDarray y_true, NDarray y_pred)
        {
            metric = call(y_true, y_pred);
            return metric;
        }

        internal abstract double call(NDarray y_true, NDarray y_pred);

        public override string ToString()
        {
            return $"{GetType().Name}:\t{metric:F4}";
        }
    }
}