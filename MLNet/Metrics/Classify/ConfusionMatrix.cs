using System.Text;
using Numpy;

namespace MLNet.Metrics
{
    public class ConfusionMatrix
    {
        public ConfusionMatrix(
            NDarray y_true,
            NDarray y_pred,
            int classify,
            double beta = 1)
        {
            Beta = beta;
            Classify = classify;
            parse(y_true, y_pred);
        }

        public int Classify { set; get; }

        public double Beta { set; get; }

        public int TruePositive { set; get; }

        public int FalsePositive { set; get; }

        public int TrueNagative { set; get; }

        public int FalseNagative { set; get; }

        public double Percision => 1.0 * TruePositive / (TruePositive + FalsePositive);


        public double ReCall => 1.0 * TruePositive / (TruePositive + FalseNagative);


        public double FMeasure =>
            (1.0 + Math.Pow(Beta, 2)) * Percision * ReCall / (Math.Pow(Beta, 2) * Percision + ReCall);

        private void parse(NDarray y_true, NDarray y_pred)
        {
            var classarray = np.array(Classify);

            var trueTrue = np.equal(y_true, classarray);
            var falseTrue = np.not_equal(y_true, classarray);

            var truePred = np.equal(y_pred, classarray);
            var falsePred = np.not_equal(y_pred, classarray);

            TruePositive = np.bitwise_and(trueTrue, truePred)
                .GetData<bool>()
                .Count(a => a);

            FalseNagative = np.bitwise_and(trueTrue, falsePred)
                .GetData<bool>()
                .Count(a => a);

            FalsePositive = np.bitwise_and(falseTrue, truePred)
                .GetData<bool>()
                .Count(a => a);

            TrueNagative = np.bitwise_and(falseTrue, falsePred)
                .GetData<bool>()
                .Count(a => a);
        }


        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"---\tConfusionMatrix Class:{Classify}\t---");
            str.AppendLine($"{TruePositive:D}|{FalseNagative:D}");
            str.AppendLine($"{FalsePositive:D}|{TrueNagative:D}");
            str.AppendLine("--- ---");
            str.AppendLine($"Percision\t{Percision:P2}");
            str.AppendLine($"Recall\t{ReCall:P2}");
            return str.ToString();
        }
    }
}