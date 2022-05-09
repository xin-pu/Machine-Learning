using Numpy;

namespace MLNet.Optimizers
{
    public class Adam : Optimizer
    {
        public Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.99) : base(learning_rate)
        {
            Beta1 = beta1;
            Beta2 = beta2;
        }

        public double Beta1 { protected set; get; }
        public double Beta2 { protected set; get; }

        public NDarray M { protected set; get; } = null!;

        public NDarray G { protected set; get; } = null!;


        internal override NDarray call(NDarray weight, NDarray grad, int epoch = 0)
        {
            if (epoch == 0)
            {
                M = np.zeros_like(weight);
                G = np.zeros_like(weight);
            }

            M = Beta1 * M + (1 - Beta1) * grad;
            G = Beta2 * G + (1 - Beta2) * np.square(grad);

            var m = M / (1 - Beta1);
            var g = G / (1 - Beta2);

            var delta_weight = LearningRate / np.sqrt(g + Epsilon) * m;

            return weight - delta_weight;
        }
    }
}