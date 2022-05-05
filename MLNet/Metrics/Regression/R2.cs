using MLNet.Utils;
using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     R-Squared
    ///     决定系数
    /// </summary>
    public class R2 : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var delta = y_true - y_pred;
            var mse = np.power(delta, np.array(2))
                .GetData<double>()
                .Average();
            return 1 - mse / np2.variance(y_true);
        }
    }
}