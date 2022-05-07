using AutoDiff;
using Numpy;

namespace MLNet.Utils
{
    public class term
    {
        public static Term matmulRow(NDarray x, Variable[] v)
        {
            var batchs = x.shape[0];
            var features = x.shape[1];

            if (v.Length != features)
                throw new Exception();

            var row = x[$"{0}:"].GetData<double>();
            var allTerm = row.Select((c, i) => c * v[i]);

            return TermBuilder.Sum(allTerm);
        }

        /// <summary>
        ///     防止过拟合 岭回归部分 L2约束
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        public static Term getRidgeLoss(Variable[] w)
        {
            var sum = TermBuilder.Sum(w.Select(a => TermBuilder.Power(a, 2)));
            var Ter = TermBuilder.Power(sum, 0.5);
            return Ter;
        }

        /// <summary>
        ///     防止过拟合 岭回归部分 L2约束
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        public static Term getLassoLoss(Variable[] w)
        {
            var abs = w.Select(i => TermBuilder.Power(TermBuilder.Power(i, 2), 0.5));
            var sum = TermBuilder.Sum(abs);
            return sum;
        }


        public static Term sigmoid(Term x, double weight = 1)
        {
            return 1.0 / (TermBuilder.Exp(-weight * x) + 1.0);
        }
    }
}