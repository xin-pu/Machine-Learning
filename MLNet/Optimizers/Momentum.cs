using Numpy;

namespace MLNet.Optimizers
{
    public class Momentum : Optimizer
    {
        public Momentum(double gamma = 0)
        {
            Gamma = gamma;
        }

        public double Gamma { set; get; }

        internal override NDarray call(NDarray weight, NDarray grad)
        {
            throw new NotImplementedException();
        }
    }
}