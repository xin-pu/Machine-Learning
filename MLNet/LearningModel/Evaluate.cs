using System.Text;

namespace MLNet.LearningModel
{
    public class Evaluate
    {
        public double MSE { set; get; }

        public double RMSE => Math.Sqrt(MSE);

        public double MAD { set; get; }

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