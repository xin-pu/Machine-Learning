using System.Text;
using Numpy;

namespace MLNet.LearningModel
{
    public class Metric
    {
        public Metric(NDarray y_true, NDarray y_pred)
        {
            MSE = getMSE(y_true, y_pred);
            MAD = getMAD(y_true, y_pred);
        }

        /// <summary>
        ///     Mean Square Error
        /// </summary>
        public double MSE { set; get; }

        public double RMSE => Math.Sqrt(MSE);

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
            str.AppendLine($"RMSE:\t{RMSE:P2}");
            str.AppendLine($"MAD:\t{MAD:P2}");
            return str.ToString();
        }

        private static double getMSE(NDarray y_true, NDarray y_pred)
        {
            var delta_mse = np.power(np.abs(y_pred - y_true), np.array(2));
            var mse = delta_mse.GetData<double>().Average();
            return mse;
        }

        private static double getMAD(NDarray y_true, NDarray y_pred)
        {
            var delta_abs = np.abs(y_pred - y_true);
            var mad = delta_abs.GetData<double>().Average();
            return mad;
        }
    }
}