/* ============================================================================
 * core/joint-routing.js — which loaded joint a recorded connection name drives.
 *
 * A frame is keyed by the world's own connection name; a loaded model's joints carry the
 * names its URDF declares, under whatever prefix the bundler derived from its *bodies*.
 * Those namings need not agree. A world-merged MJCF robot has bodies called
 * "montessori/link0", so its model prefix is "montessori", while its connections stay
 * bare ("joint1") or start with a slash ("/finger_joint1"); a Gazebo world prefixes both.
 * So a key is tried as a prefixed name first -- that is what keeps two models declaring
 * the same joint name apart -- and then as a plain joint name across every model.
 * ==========================================================================*/
(function (global) {
  'use strict';

  function jointNamed(model, name) {
    const joints = model.obj && model.obj.joints;
    return (joints && joints[name]) || null;
  }

  function anyJointNamed(models, name) {
    for (let index = 0; index < models.length; index += 1) {
      const joint = jointNamed(models[index], name);
      if (joint) return joint;
    }
    return null;
  }

  //: the URDF joint types a user may set by hand
  const MOVABLE_TYPES = { revolute: true, prismatic: true, continuous: true };

  //: the key the world names a model's joint by: prefixed after the model, or bare
  function keyOf(model, name) {
    return model.prefix ? model.prefix + '/' + name : name;
  }

  global.JointRouting = {
    /* The joints of the environment models a user may move by hand -- doors, drawers,
       turntables -- each as {key, name, joint, model}, in declaration order. The robot's
       own joints are the plan's to drive, so a model flagged `robot` contributes none. */
    movableJoints: function (models) {
      const found = [];
      models.forEach(function (model) {
        if (model.robot) return;
        const joints = (model.obj && model.obj.joints) || {};
        Object.keys(joints).forEach(function (name) {
          const joint = joints[name];
          if (!MOVABLE_TYPES[joint.jointType]) return;
          found.push({ key: keyOf(model, name), name: name, joint: joint, model: model });
        });
      });
      return found;
    },

    /* The positions `joint` may take, as {lower, upper}: its URDF limits, or a full turn
       for a continuous joint, which declares none. */
    range: function (joint) {
      if (joint.jointType === 'continuous') return { lower: -Math.PI, upper: Math.PI };
      return { lower: joint.limit.lower, upper: joint.limit.upper };
    },

    /* The joint `key` drives, or null when no loaded model declares it.

       `models` are the loaded model entries the shell holds: a `prefix` and the URDF's
       joints under `obj.joints`. */
    jointFor: function (models, key) {
      // a leading slash is part of the name, not an empty prefix
      const cut = key.indexOf('/');
      if (cut > 0) {
        const prefix = key.slice(0, cut);
        const name = key.slice(cut + 1);
        for (let index = 0; index < models.length; index += 1) {
          if (models[index].prefix !== prefix) continue;
          const joint = jointNamed(models[index], name);
          if (joint) return joint;
        }
        return anyJointNamed(models, key) || anyJointNamed(models, name);
      }
      return anyJointNamed(models, key);
    },
  };
})(window);
