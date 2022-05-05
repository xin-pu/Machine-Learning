using System.Text;
using MLNet.Utils;
using Numpy;

namespace MLNet.LearningModel
{
    public class Metric
    {
        public Metric(NDarray y_true, NDarray y_pred)
        {
            MSE = getMSE(y_true, y_pred);
            MAD = getMAD(y_true, y_pred);
            R2 = getR2(y_true, y_pred);
            EVS = getEVS(y_true, y_pred);
        }

        /// <summary>
        ///     Mean Square Error
        /// </summary>
        public double MSE { set; get; }


        /// <summary>
        ///     Mean Absolute Error
        /// </summary>
        public double MAD { set; get; }

        /// <summary>
        ///     R2 score
        /// </summary>
        public double R2 { set; get; }

        /// <summary>
        ///     Explained variance score
        /// </summary>
        public double EVS { set; get; }


        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine(new string('-', 30) + "Evaluate" + new string('-', 30));
            str.AppendLine($"MSE:\t{MSE:P2}");
            str.AppendLine($"MAD:\t{MAD:P2}");
            str.AppendLine($"EVS:\t{EVS:P2}");
            str.AppendLine($"R2:\t{R2:P2}");
            return str.ToString();
        }

        public static double getMSE(NDarray y_true, NDarray y_pred)
        {
            var delta_mse = np.power(np.abs(y_pred - y_true), np.array(2));
            var mse = delta_mse.GetData<double>().Average();
            return mse;
        }

        public static double getMAD(NDarray y_true, NDarray y_pred)
        {
            var delta_abs = np.abs(y_pred - y_true);
            var mad = delta_abs.GetData<double>().Average();
            return mad;
        }

        public static double getR2(NDarray y_true, NDarray y_pred)
        {
            return 0;
        }

        public static double getEVS(NDarray y_true, NDarray y_pred)
        {
            var delta = y_true - y_pred;
            var varuance_with_pred = np2.variance(delta);

            var variance_y_true = np2.variance(y_true);
            return 1 - varuance_with_pred / variance_y_true;
        }
    }
}