using AutoDiff;
using MLNet.Loss;
using MLNet.Utils;
using Numpy;

namespace MLNet.Model.Classify
{
    public class BinaryLogicClassify : Model
    {
        public BinaryLogicClassify()
        {
            Name = "BinaryLogicClassify";
        }

        internal override NDarray convert(NDarray x)
        {
            return np2.linear_first_order(x);
        }

        internal override void fit(NDarray x, NDarray y, double learning_rate, int epoch)
        {
            var featureCount = x.shape[1];

            var resolveTemp = np.random.randn(featureCount, 1);

            Enumerable.Range(0, epoch).ToList().ForEach(e =>
            {
                var theda = resolveTemp.GetData<double>();

                var loss = CostFunc.Evaluate(theda);
                var gradarray = CostFunc.Gradient(theda);

                var grad = np.expand_dims(np.array(gradarray), -1);
                resolveTemp -= learning_rate * grad;

                if (Print)
                    Log.print?.Invoke($"Epoch:\t{e:D5}\tLoss:{loss:F4}");
            });
            Resolve = resolveTemp;
        }

        internal override NDarray call(NDarray x)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            return np2.sigmoid(np.matmul(x, Resolve));
        }

        internal override LossBase initialLoss(Variable[] variables, NDarray x, NDarray y)
        {
            return new CrossEntropy("CrossEntropy", variables, x, y);
        }
    }
}