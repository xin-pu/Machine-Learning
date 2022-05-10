using Numpy;

namespace MLNet.Optimizers
{
    public class RMSProp : Optimizer
    {
        public RMSProp(double workLearningRate, double beta = 0.9)
            : base(workLearningRate)
        {
            Beta = beta;
        }

        /// <summary>
        ///     衰减率
        /// </summary>
        public double Beta { protected set; get; }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        public NDarray G { set; get; } = null!;

        internal override NDarray call(NDarray weight, NDarray grad, int epoch)
        {
            if (epoch == 0)
                G = np.zeros_like(weight);

            G = Beta * G + (1 - Beta) * np.square(grad);
            var delta = -np.multiply(WorkLearningRate / np.sqrt(G + epsilon), grad);
            return weight + delta;
        }
    }
}