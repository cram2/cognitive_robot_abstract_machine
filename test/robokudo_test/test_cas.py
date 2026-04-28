import robokudo.cas
import robokudo.types.annotation
import robokudo.types.scene


class TestCASObject(object):
    cls = robokudo.cas.CAS

    def test_annotations(self):
        cas = self.cls()

        classification = robokudo.types.annotation.Classification()
        classification.source = "classifier"
        color1 = robokudo.types.annotation.SemanticColor()
        color1.color = "black"
        color2 = robokudo.types.annotation.SemanticColor()
        color2.color = "red"

        rs_obj = robokudo.types.scene.ObjectHypothesis()
        rs_obj.annotations.append(classification)
        rs_obj.annotations.append(color1)
        rs_obj.annotations.append(color2)

        classification_list = cas.filter_by_type(
            robokudo.types.annotation.Classification, rs_obj.annotations
        )
        color_list = cas.filter_by_type(
            robokudo.types.annotation.SemanticColor, rs_obj.annotations
        )

        assert len(classification_list) == 1
        assert len(color_list) == 2
        assert color_list[0] == color1
        assert color_list[1] == color2

    def test_ohs_with_criteria(self):
        cas = self.cls()

        classification = robokudo.types.annotation.Classification()
        classification.source = "classifier"
        classification.classname = "Spoon"
        color1 = robokudo.types.annotation.SemanticColor()
        color1.color = "black"
        color2 = robokudo.types.annotation.SemanticColor()
        color2.color = "red"

        rs_obj = robokudo.types.scene.ObjectHypothesis()
        rs_obj.annotations.append(classification)
        rs_obj.annotations.append(color1)
        rs_obj.annotations.append(color2)
        rs_obj2 = robokudo.types.scene.ObjectHypothesis()

        cl = cas.filter_by_type_and_criteria(
            robokudo.types.annotation.Classification,
            rs_obj.annotations,
            {"classname": ("==", "Spoon")},
        )
        assert len(cl) == 1

        # Attribute checking is case-sensitive!
        cl = cas.filter_by_type_and_criteria(
            robokudo.types.annotation.Classification,
            rs_obj2.annotations,
            {"classname": ("==", "spoon")},
        )
        assert len(cl) == 0

        # Other Object without annotations
        cl = cas.filter_by_type_and_criteria(
            robokudo.types.annotation.Classification,
            rs_obj2.annotations,
            {"classname": ("==", "Spoon")},
        )
        assert len(cl) == 0

        cl = cas.filter_by_type_and_criteria(
            robokudo.types.annotation.SemanticColor,
            rs_obj.annotations,
            {"color": ("==", "red")},
        )
        assert len(cl) == 1
