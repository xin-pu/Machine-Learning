using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     Precision
    ///     精确率 或精度 或查准率
    /// </summary>
    public class Precision : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            throw new NotImplementedException();
        }
    }
}