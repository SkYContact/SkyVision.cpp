// backend_factories.hpp - backend factory declarations.
//
// Included only by vqa_registry.cpp. Keeping these out of vqa.hpp means the
// public interface header stays free of the backend list: adding a backend is
// one declaration here plus one row in vqa_registry.cpp's arch table.
#pragma once

#include "core/vqa.hpp"

namespace vqa {

Model * create_moondream();
Model * create_videochat3();
Model * create_qwen3_5();
Model * create_moss_vl();
Model * create_videomae();
Model * create_vjepa2();
Model * create_qwen2vl();
Model * create_smolvlm();
Model * create_mage_vl();
Model * create_qwen25omni();
Model * create_embedgemma2();
Model * create_siglip();

}  // namespace vqa
