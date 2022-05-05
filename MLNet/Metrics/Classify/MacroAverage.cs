using Numpy;

namespace MLNet.Metrics
{
    public class MacroReCall : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var con = new ConfusionMatrixs(y_true, y_pred);
            var res = con.ConfusionMatrixDict
                .Select(a => a.Value.ReCall)
                .Average();
            return res;
        }
    }

    public class MacroPercision : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var con = new ConfusionMatrixs(y_true, y_pred);
            var res = con.ConfusionMatrixDict
                .Select(a => a.Value.Percision)
                .Average();
            return res;
        }
    }

    public class MacroFMeasure : Metric
    {
        internal override double call(NDarray y_true, NDarray y_pred)
        {
            var con = new ConfusionMatrixs(y_true, y_pred);
            var res = con.ConfusionMatrixDict
                .Select(a => a.Value.FMeasure)
                .Average();
            return res;
        }
    }
}