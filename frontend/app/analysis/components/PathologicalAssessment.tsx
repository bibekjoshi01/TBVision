import { ClipboardList } from "lucide-react";
import { Input } from "@heroui/input";
import { Select as UISelect, SelectItem as UISelectItem } from "@heroui/select";
import { Switch as UISwitch } from "@heroui/switch";
import { Divider } from "@heroui/divider";
import { Controller } from "react-hook-form";

const Select = UISelect as any;
const SelectItem = UISelectItem as any;
const Switch = UISwitch as any;

interface PathologicalAssessmentProps {
  control: any;
}

export function PathologicalAssessment({ control }: PathologicalAssessmentProps) {
  return (
    <div className="space-y-6">
      <p className="text-xs font-bold text-default-600 uppercase tracking-widest flex items-center gap-2">
        <ClipboardList size={14} /> 3. Pathological Assessment
      </p>
      <div className="space-y-10">
        {/* Primary Symptoms */}
        <div className="space-y-4">
          <p className="text-sm font-bold text-default-600 uppercase">Primary Symptoms</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-3">
            {["cough", "fever", "night_sweats", "weight_loss", "chest_pain", "shortness_of_breath", "fatigue"].map((s) => (
              <Controller
                key={s}
                name={`symptoms.${s}` as any}
                control={control}
                render={({ field }) => (
                  <Switch isSelected={field.value} onValueChange={field.onChange} size="sm">
                    <span className="text-sm font-medium text-default-900 capitalize">{s.replace("_", " ")}</span>
                  </Switch>
                )}
              />
            ))}
          </div>
          <div className="pt-2">
            <Controller
              name="symptoms.cough_duration_days"
              control={control}
              render={({ field }) => (
                <Input {...field} type="number" label="Cough Duration (Days)" variant="bordered" size="md" className="max-w-xs" />
              )}
            />
          </div>
        </div>

        <Divider />

        {/* Risk Presence */}
        <div className="space-y-4">
          <p className="text-sm font-bold text-default-600 uppercase">Risk Presence</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-3">
            {["smoker", "diabetes", "hiv_positive", "close_contact_tb_patient", "immunocompromised"].map((r) => (
              <Controller
                key={r}
                name={`risk_factors.${r}` as any}
                control={control}
                render={({ field }) => (
                  <Switch isSelected={field.value} onValueChange={field.onChange} size="sm">
                    <span className="text-sm font-medium text-default-900 capitalize">{r.split("_").join(" ")}</span>
                  </Switch>
                )}
              />
            ))}
          </div>
        </div>

        <Divider />

        {/* Medical History */}
        <div className="space-y-4">
          <p className="text-sm font-bold text-default-600 uppercase">Medical History</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-3">
            {["previous_tb", "chronic_lung_disease", "recent_pneumonia"].map((h) => (
              <Controller
                key={h}
                name={`medical_history.${h}` as any}
                control={control}
                render={({ field }) => (
                  <Switch isSelected={field.value} onValueChange={field.onChange} size="sm" className="w-full">
                    <span className="text-sm font-medium text-default-900 capitalize">{h.split("_").join(" ")}</span>
                  </Switch>
                )}
              />
            ))}
          </div>
        </div>

        <Divider />

        {/* Screening Context */}
        <div className="space-y-6">
          <p className="text-sm font-bold text-default-600 uppercase">Screening Context</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Controller
              name="screening_context.screening_type"
              control={control}
              render={({ field }) => (
                <Select {...field} label="Screening Type" variant="bordered" size="md" selectedKeys={[field.value]} onSelectionChange={(keys: any) => field.onChange(Array.from(keys)[0])}>
                  <SelectItem key="symptomatic">Symptomatic</SelectItem>
                  <SelectItem key="preventive">Preventive</SelectItem>
                  <SelectItem key="occupational">Occupational</SelectItem>
                </Select>
              )}
            />
            <Controller
              name="screening_context.location"
              control={control}
              render={({ field }) => (
                <Input {...field} label="Clinical Location" variant="bordered" size="md" />
              )}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
