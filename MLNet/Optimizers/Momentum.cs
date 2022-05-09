using Numpy;

namespace MLNet.Optimizers
{
    public class Momentum : Optimizer
    {
        public Momentum(double learningrate, double gamma = 0.9)
            : base(learningrate)
        {
            Gamma = gamma;
        }

        public double Gamma { set; get; }

        public NDarray DeltaTheda { protected set; get; } = null!;

        internal override NDarray call(NDarray weight, NDarray grad, int epoch = 0)
        {
            if (epoch == 0)
                DeltaTheda = np.zeros_like(weight);

            DeltaTheda = Gamma * DeltaTheda - LearningRate * grad;

            return weight + DeltaTheda;
        }
    }
}