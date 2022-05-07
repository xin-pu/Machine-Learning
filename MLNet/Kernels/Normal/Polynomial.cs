using Numpy;

namespace MLNet.Kernels
{
    /// <summary>
    ///     input:[x]
    ///     output: [1,x,x^2,x^3,...,x^N]
    /// </summary>
    public class Polynomial : Transform
    {
        public Polynomial(int degree)
        {
            Degree = degree;
        }

        public int Degree { protected set; get; }


        public override NDarray Call(NDarray input)
        {
            var batch = input.shape[0];
            var features = input.shape[1];
            if (features != 1) throw new Exception("Regression for 1 dims");

            var xTranspose = np.transpose(input);
            var npX = np.ones(Degree + 1, batch);
            Enumerable.Range(1, Degree).ToList().ForEach(d =>
            {
                var row = np.ones(input.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return npX;
        }
    }
}