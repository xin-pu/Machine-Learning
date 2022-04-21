using System.Text;

namespace MLNet.LearningModel
{
    public class Metric
    {
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
    }
}