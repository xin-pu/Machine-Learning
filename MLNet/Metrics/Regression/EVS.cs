using MLNet.Utils;
using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     Explained variance
    ///     可解释变异
    /// </summary>
    public class EVS : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var delta = y_true - y_pred;
            var varuance_with_pred = np2.variance(delta);

            var variance_y_true = np2.variance(y_true);
            return 1 - varuance_with_pred / variance_y_true;
        }
    }
}