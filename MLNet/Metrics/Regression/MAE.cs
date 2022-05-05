using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     Mean Absolute Error
    ///     平均绝对值误差
    /// </summary>
    public class MAE : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var delta_abs = np.abs(y_pred - y_true);
            var mae = delta_abs.GetData<double>().Average();
            return mae;
        }
    }
}