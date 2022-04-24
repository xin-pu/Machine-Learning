using Numpy;

namespace MLNet.Kernels
{
    public abstract class Kernel
    {
        protected Kernel(KernelType kernelType)
        {
            KernelType = kernelType;
        }

        public KernelType KernelType { protected set; get; }

        public abstract NDarray Transform(NDarray input);
    }
}