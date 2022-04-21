﻿using AutoDiff;
using MLNet.Loss;
using MLNet.Utils;
using Numpy;

namespace MLNet.Model.Regression.LinearRegression
{
    /// <summary>
    ///     多元线性回归
    ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
    /// </summary>
    public class MultipleLinearRegression : Model
    {
        public MultipleLinearRegression(string name = "MultipleLinearRegression",
            Constraint constraint = Constraint.None)
            : base(name)
        {
            Constraint = constraint;
        }

        public Constraint Constraint { set; get; }

        public override LossBase CostFunc { get; set; } = null!;
        public override NDarray Resolve { get; set; } = null!;

        public NDarray Slove(NDarray x, NDarray y)
        {
            return np.linalg.pinv(x).dot(y);
        }


        internal NDarray slove(NDarray x, NDarray y)
        {
            return np.linalg.pinv(x).dot(y);
        }

        /// <summary>
        ///     x => 1,x1,x2,x3,...,xN
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal override NDarray convert(NDarray x)
        {
            return np2.linear_first_order(x);
        }

        internal override void fit(NDarray x, NDarray y, double learning_rate, int epoch)
        {
            var featureCount = x.shape[1];

            var resolve = np.random.randn(featureCount, 1);

            Enumerable.Range(0, epoch).ToList().ForEach(e =>
            {
                var theda = resolve.GetData<double>();

                var loss = CostFunc.Evaluate(theda);
                var gradarray = CostFunc.Gradient(theda);

                var grad = np.expand_dims(np.array(gradarray), -1);
                resolve -= learning_rate * grad;

                if (Print)
                    Log.print?.Invoke($"{Name} Epoch:\t{e:D5}\tLoss:{loss:F4}");
            });
            Resolve = resolve;
        }


        internal override NDarray call(NDarray x)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            return np.matmul(x, Resolve);
        }

        internal override LossBase initialLoss(Variable[] variables, NDarray x, NDarray y)
        {
            return new LSLoss(variables, x, y)
            {
                Constraint = Constraint,
                Lamdba = 0.1
            };
        }
    }
}