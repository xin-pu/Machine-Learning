﻿using Numpy;

namespace MLNet.Optimizers
{
    /// <summary>
    /// </summary>
    public class AdaGrad : Optimizer
    {
        public AdaGrad(double learning_rate)
            : base(learning_rate)
        {
        }

        /// <summary>
        ///     参数梯度平方的累计值
        /// </summary>
        public NDarray G { protected set; get; } = null!;

        internal override NDarray call(NDarray weight, NDarray grad, int epoch)
        {
            if (epoch == 0)
                G = np.zeros_like(weight);

            G += np.square(grad);
            var delta = -np.multiply(LearningRate / np.sqrt(G + epsilon), grad);
            return weight + delta;
        }
    }
}