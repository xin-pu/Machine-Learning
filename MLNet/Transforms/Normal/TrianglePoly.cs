using Numpy;

namespace MLNet.Transforms
{
    /// <summary>
    ///     input:  [x]
    ///     output: [1,sin(x/2),cos(x/2),sin(2x/x),cos(2x/2)...sin(degree*x/2),cos(degree*x/2)]
    /// </summary>
    public class TrianglePoly : Transform
    {
        public TrianglePoly(int degree)
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
            var npX = np.ones(2 * Degree + 1, batch);
            Enumerable.Range(0, Degree).ToList().ForEach(d =>
            {
                npX[1 + 2 * d] = np.sin(d * xTranspose / 2);
                npX[2 + 2 * d] = np.cos(d * xTranspose / 2);
            });
            npX = np.transpose(npX);
            return npX;
        }
    }
}