using Numpy;

namespace MLNet.Kernels
{
    /// <summary>
    ///     input: [x1,x2,x3,...,xN]
    ///     output: [1,x1,x2,x3,...,xN]
    /// </summary>
    public class LinearFirstorder : Transform
    {
        public override NDarray Call(NDarray input)
        {
            var b = np.ones(input.shape[0]);
            var res = np.insert(input, 0, b, 1);
            return res;
        }
    }
}