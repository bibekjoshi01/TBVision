import { User } from "lucide-react";
import { Input } from "@heroui/input";
import { Select as UISelect, SelectItem as UISelectItem } from "@heroui/select";
import { Controller } from "react-hook-form";

const Select = UISelect as any;
const SelectItem = UISelectItem as any;

interface PatientDemographicsProps {
  control: any;
  errors: any;
}

export function PatientDemographics({ control, errors }: PatientDemographicsProps) {
  return (
    <div className="space-y-6">
      <p className="text-xs font-bold text-default-600 uppercase tracking-widest flex items-center gap-2">
        <User size={14} /> 2. Patient Demographics
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <Controller
          name="patient_info.age"
          control={control}
          render={({ field }) => (
            <Input {...field} type="number" label="Age" variant="bordered" size="md" isInvalid={!!errors.patient_info?.age} />
          )}
        />
        <Controller
          name="patient_info.sex"
          control={control}
          render={({ field }) => (
            <Select {...field} label="Sex" variant="bordered" size="md" selectedKeys={[field.value]} onSelectionChange={(keys: any) => field.onChange(Array.from(keys)[0])}>
              <SelectItem key="male">Male</SelectItem>
              <SelectItem key="female">Female</SelectItem>
              <SelectItem key="other">Other</SelectItem>
            </Select>
          )}
        />
        <Controller
          name="patient_info.region"
          control={control}
          render={({ field }) => (
            <Select {...field} label="Region" variant="bordered" size="md" selectedKeys={[field.value || ""]} onSelectionChange={(keys: any) => field.onChange(Array.from(keys)[0])}>
              <SelectItem key="South Asia">South Asia</SelectItem>
              <SelectItem key="Southeast Asia">Southeast Asia</SelectItem>
              <SelectItem key="Sub-Saharan Africa">Sub-Saharan Africa</SelectItem>
              <SelectItem key="East Asia">East Asia</SelectItem>
              <SelectItem key="Middle East">Middle East</SelectItem>
              <SelectItem key="Europe">Europe</SelectItem>
              <SelectItem key="Americas">Americas</SelectItem>
              <SelectItem key="Oceania">Oceania</SelectItem>
            </Select>
          )}
        />
      </div>
    </div>
  );
}
