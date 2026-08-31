// Custom SIREN (Sinusoidal Representation Network) layer for TensorFlow.js.
//
// `sin` is NOT a built-in dense activation in tf.js, and SIREN additionally
// requires a very specific weight initialization:
//   - first layer: uniform in ±1/fan_in
//   - subsequent layers: uniform in ±√(6/fan_in)/ω₀, with ω₀ = 30
//
// Without this init, SIREN visibly fails to train (the sine activations stay
// stuck in flat/linear regions). We provide both a plain sine Activation
// class (wrapped in a lambda Dense layer) and a custom SIREN Dense layer
// that bakes in the correct fan_in-based initialization.

const SIREN_W0 = 30;

function uniformRandom(min, max) {
  return () => Math.random() * (max - min) + min;
}

// A tf.js Activation that applies sin(x). Used inside a lambda-wrapped dense
// layer so we get the Dense layout/param bookkeeping with a custom activation.
class SineActivation extends tf.layers.Activation {
  static className = 'SineActivation';

  apply(x) {
    return tf.tidy(() => tf.sin(x));
  }
}
tf.serialization.registerClass(SineActivation);

// Custom Dense layer with SIREN-style initialization baked into build().
class SirenDense extends tf.layers.Layer {
  static className = 'SirenDense';
  constructor(config) {
    super(config);
    this.units = config.units;
    this.isFirstLayer = config.isFirstLayer != null ? config.isFirstLayer : false;
    this.w0 = config.w0 != null ? config.w0 : SIREN_W0;
    this.useBias = config.useBias != null ? config.useBias : true;
    this.kernel = null;
    this.bias = null;
  }

  build(inputShape) {
    const fanIn = inputShape[inputShape.length - 1];
    const fanOut = this.units;

    // SIREN init: first layer -> ±1/fan_in, deeper -> ±√(6/fan_in)/ω₀.
    // We can't know depth here, so we store the tailored init via config.
    const bound = this.isFirstLayer
      ? 1 / Math.max(1, fanIn)
      : Math.sqrt(6 / Math.max(1, fanIn)) / this.w0;

    const kernelInit = tf.initializers.randomUniform({
      minval: -bound,
      maxval: bound,
    });

    this.kernel = this.addWeight(
      'kernel',
      [fanIn, this.units],
      'float32',
      kernelInit
    );
    if (this.useBias) {
      this.bias = this.addWeight(
        'bias',
        [this.units],
        'float32',
        tf.initializers.zeros()
      );
    }
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      let out = x.matMul(this.kernel.read());
      if (this.bias) out = out.add(this.bias.read());
      // SIREN's defining nonlinearity: sin(ω₀ · z). In the original paper,
      // ω₀ scales the FIRST layer's output; hidden layers use ω₀=1 by default,
      // but applying the same scale everywhere is a common, simpler variant
      // that works well here.
      return tf.sin(tf.mul(out, this.w0));
    });
  }

  computeOutputShape(inputShape) {
    const shape = inputShape.slice();
    shape[shape.length - 1] = this.units;
    return shape;
  }

  getConfig() {
    const config = super.getConfig();
    config.units = this.units;
    config.isFirstLayer = this.isFirstLayer;
    config.w0 = this.w0;
    config.useBias = this.useBias;
    return config;
  }
}
tf.serialization.registerClass(SirenDense);

// Build a SIREN Dense layer. The `isFirstLayer` flag selects the correct
// initialization bound (uniform ±1/fan_in for first, ±√(6/fan_in)/ω₀ after).
function sirenDense(units, isFirstLayer, w0) {
  return new SirenDense({
    units,
    w0: w0 != null ? w0 : SIREN_W0,
    isFirstLayer: !!isFirstLayer,
  });
}
