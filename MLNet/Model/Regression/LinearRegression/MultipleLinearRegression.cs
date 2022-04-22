using AutoDiff;
using MLNet.Loss;
using Numpy;

namespace MLNet.Model.Regression.LinearRegression
{
    /// <summary>
    ///     多元线性回归
    ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
    /// </summary>
    public class MultipleLinearRegression : Model
    {
        public MultipleLinearRegression()
            : base("MultipleLinearRegression")
        {
        }

        public MultipleLinearRegression(string name = "MultipleLinearRegression",
            Constraint constraint = Constraint.None)
            : base(name)
        {
            Constraint = constraint;
        }

        public Constraint Constraint { set; get; }


        /// <summary>
        ///     x => 1,x1,x2,x3,...,xN
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal override NDarray transform(NDarray x)
        {
            return Utils.transform.to_linear_firstorder(x);
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
                    Log.print?.Invoke($"{Name} Epoch:\t{e:D5}\tLoss:{loss:F4}");
            });
            Resolve = resolveTemp;
        }


        internal override NDarray call(NDarray x)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            return np.matmul(x, Resolve);
        }

        internal override LossBase initialLoss(
            Variable[] variables,
            NDarray x,
            NDarray y)
        {
            return new LSLoss(variables, x, y)
            {
                Constraint = Constraint,
                Lamdba = 0.1
            };
        }
    }
}