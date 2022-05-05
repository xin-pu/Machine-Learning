using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     Accuracy
    ///     准确率
    /// </summary>
    public class Accuracy : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var res = np.equal(y_pred, y_true);
            var tptn = res.GetData<bool>().Count(a => a);
            return 1.0 * tptn / y_true.len;
        }
    }
}