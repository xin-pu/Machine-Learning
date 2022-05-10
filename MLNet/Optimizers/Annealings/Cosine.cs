namespace MLNet.Optimizers
{
    /// <summary>
    ///     余弦衰减
    /// </summary>
    public class Cosine : Annealing
    {
        public Cosine(double learningrate, int totalepoch) : base(learningrate)
        {
            TotalEpoch = totalepoch;
        }

        public int TotalEpoch { protected get; set; }

        internal override double UpdateLearningRate(int epoch)
        {
            return 0.5 * InitLearningRate * (1 + Math.Cos(epoch * Math.PI / TotalEpoch));
        }
    }
}