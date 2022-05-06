using AutoDiff;
using MLNet.Kernels;
using MLNet.Losses;
using MLNet.Utils;
using Numpy;

namespace MLNet.Models.Classify
{
    public class BinaryLogicClassify : Model
    {
        public BinaryLogicClassify()
        {
            Name = "BinaryLogicClassify";
        }

        internal override NDarray transform(NDarray x)
        {
            return new Gaussian(2).Transform(x);
        }

        internal override void fit(NDarray x, NDarray y, double learning_rate, int epoch, int batchsize)
        {
            var featureCount = x.shape[1];

            var resolveTemp = np.random.randn(featureCount, 1);

            Enumerable.Range(0, epoch).ToList().ForEach(e =>
            {
                var (gradarray, loss) = CostFunc.Call(resolveTemp);

                var grad = np.expand_dims(np.array(gradarray), -1);
                resolveTemp -= learning_rate * grad;

                if (Print)
                    Log.print?.Invoke($"Epoch:\t{e:D4}\tLoss:{loss:F4}");
            });
            Resolve = resolveTemp;
        }

        internal override NDarray call(NDarray x)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            return np2.sigmoid(np.matmul(x, Resolve));
        }

        internal override Loss initialLoss(Variable[] variables, NDarray x, NDarray y)
        {
            return new CrossEntropy(variables, x, y);
        }
    }
}