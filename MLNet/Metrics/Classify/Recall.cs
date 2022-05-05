using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     Recall
    ///     召回率，查全率
    /// </summary>
    public class Recall : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            throw new NotImplementedException();
        }
    }
}