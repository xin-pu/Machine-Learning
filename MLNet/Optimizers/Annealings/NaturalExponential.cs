namespace MLNet.Optimizers
{
    public class NaturalExponential : Annealing
    {
        public NaturalExponential(double learningrate, double beta = 0.04)
            : base(learningrate)
        {
            Beta = beta;
        }

        public double Beta { protected set; get; }

        internal override double UpdateLearningRate(int epoch)
        {
            return InitLearningRate * Math.Exp(-Beta * epoch);
        }
    }
}