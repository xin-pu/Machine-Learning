using System.Text;

namespace MLNet.LearningModel
{
    public class Evaluate
    {
        public double MSE { set; get; }

        public double MAD { set; get; }

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"MSE:\t{MSE:P2}");
            str.AppendLine($"MAD:\t{MAD:P2}");
            return str.ToString();
        }
    }
}