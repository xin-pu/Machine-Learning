using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     ErrorRate
    ///     错误率
    /// </summary>
    public class ErrorRate : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var res = np.equal(y_pred, y_true);
            var tptn = res.GetData<bool>().Count(a => a);
            return 1 - 1.0 * tptn / y_true.len;
        }
    }
}