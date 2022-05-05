using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     FMeasure
    ///     精确率和召回率的调和平均
    /// </summary>
    public class FMeasure : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            throw new NotImplementedException();
        }
    }
}