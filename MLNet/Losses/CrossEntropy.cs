using AutoDiff;
using MLNet.Utils;
using Numpy;

namespace MLNet.Losses
{
    public class CrossEntropy : Loss
    {
        public CrossEntropy(Variable[] variables, NDarray x, NDarray y)
            : base("CrossEntropy", variables, x, y)
        {
        }

        internal override Term createLoss(Variable[] variables, NDarray x, NDarray y)
        {
            var batchsize = x.shape[0];
            var list = Enumerable.Range(0, batchsize).Select(i =>
            {
                var rowX = x[$"{i}:{i + 1},:"];
                var yp = y[$"{i},:"].GetData<double>()[0];
                var xp = term.matmul(rowX, variables);
                var xp_sigmoid = term.sigmoid(xp);
                var fin = yp * TermBuilder.Log(xp_sigmoid) + (1 - yp) * TermBuilder.Log(1 - xp_sigmoid);
                return fin;
            });

            return -TermBuilder.Sum(list) / batchsize;
        }
    }
}