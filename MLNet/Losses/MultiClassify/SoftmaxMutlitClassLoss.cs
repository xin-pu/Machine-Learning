using AutoDiff;
using MLNet.Utils;
using Numpy;

namespace MLNet.Losses
{
    public class SoftmaxMutlitClassLoss : MultiClassLoss
    {
        public SoftmaxMutlitClassLoss(int classes)
            : base(classes)
        {
        }


        internal override Term createLoss(Dictionary<int, Variable[]> w, NDarray x, NDarray y)
        {
            var batchsize = x.shape[0];
            var list = Enumerable.Range(0, batchsize).Select(i =>
            {
                var rowX = x[$"{i}:{i + 1},:"];
                var yp = y[$"{i},:"].GetData<int>()[0];
                var xp = term.matmulRow(rowX, w[yp]);
                var molecule = TermBuilder.Exp(xp);
                var denominator = TermBuilder.Sum(w.Select(a => TermBuilder.Exp(term.matmulRow(rowX, a.Value))));
                return molecule / denominator;
            });

            return -TermBuilder.Sum(list) / batchsize;
        }
    }
}