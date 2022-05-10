namespace MLNet.Optimizers
{
    public class InverseTime : LRSchedule
    {
        /// <summary>
        ///     逆时衰减
        /// </summary>
        /// <param name="learningrate"></param>
        /// <param name="beta"></param>
        public InverseTime(double learningrate, double beta = 0.1)
            : base(learningrate)
        {
            Beta = beta;
        }

        public double Beta { protected set; get; }

        internal override double UpdateLearningRate(int epoch)
        {
            return InitLearningRate / (1 + Beta * epoch);
        }
    }
}